use std::fs::File;
use std::io::BufReader;

use ::e57::{CartesianCoordinate, E57Reader, RecordDataType, RecordName, RecordValue};
use numpy::ndarray::Array2;
use numpy::{IntoPyArray, PyArray1, PyArray2};
use pyo3::exceptions::{PyIndexError, PyRuntimeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::PyDict;

#[pyclass]
pub struct E57 {
    #[pyo3(get)]
    pub points: Py<PyArray2<f64>>,
    #[pyo3(get)]
    pub color: Py<PyArray2<f32>>,
    #[pyo3(get)]
    pub intensity: Py<PyArray2<f32>>,
}

/// Extracts the xml contents from an e57 file.
#[pyfunction]
fn raw_xml(filepath: &str) -> PyResult<String> {
    let file = File::open(filepath)?;
    let reader = BufReader::new(file);
    let xml = E57Reader::raw_xml(reader);
    let xml = match &xml {
        Ok(_) => xml,
        Err(e) => {
            return Err(PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(
                e.to_string(),
            ));
        }
    };
    let xml_string = String::from_utf8(xml.unwrap())?;
    Ok(xml_string)
}

/// Extracts the point data from an e57 file.
#[pyfunction]
unsafe fn read_points(py: Python<'_>, filepath: &str) -> PyResult<E57> {
    let file = E57Reader::from_file(filepath);
    let mut file = match file {
        Ok(file) => file,
        Err(e) => {
            return Err(PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(
                e.to_string(),
            ));
        }
    };
    let pc = file.pointclouds();
    let pc = pc.first().expect("files contain pointclouds");
    let mut point_vec = Vec::with_capacity(pc.records as usize * 3);
    let mut color_vec = Vec::with_capacity(pc.records as usize * 3);
    let mut intensity_vec = Vec::with_capacity(pc.records as usize);
    let mut nrows = 0;
    for pointcloud in file.pointclouds() {
        let mut iter = file
            .pointcloud_simple(&pointcloud)
            .expect("this file should contain a pointcloud");
        iter.spherical_to_cartesian(true);
        iter.cartesian_to_spherical(false);
        iter.intensity_to_color(true);
        iter.normalize_intensity(false);
        iter.apply_pose(true);

        for p in iter {
            let p = p.expect("Unable to read next point");
            if let CartesianCoordinate::Valid { x, y, z } = p.cartesian {
                point_vec.extend([x, y, z]);
                nrows += 1
            }
            if let Some(color) = p.color {
                color_vec.extend([color.red, color.green, color.blue])
            }
            if let Some(intensity) = p.intensity {
                intensity_vec.push(intensity);
            }
        }
    }
    let n_points = point_vec.len() / 3;
    let n_colors = color_vec.len() / 3;
    let n_intensities = intensity_vec.len();
    let mut e57 = E57 {
        points: Array2::from_shape_vec((nrows, 3), point_vec)
            .unwrap()
            .into_pyarray(py)
            .unbind(),
        color: Array2::<f32>::zeros((0, 3)).into_pyarray(py).unbind(),
        intensity: Array2::<f32>::zeros((0, 1)).into_pyarray(py).unbind(),
    };
    if n_colors == n_points {
        e57.color = Array2::from_shape_vec((nrows, 3), color_vec)
            .unwrap()
            .into_pyarray(py)
            .unbind();
    }
    if n_intensities == n_points {
        e57.intensity = Array2::from_shape_vec((nrows, 1), intensity_vec)
            .unwrap()
            .into_pyarray(py)
            .unbind();
    }
    Ok(e57)
}

fn runtime_error(e: impl std::fmt::Display) -> PyErr {
    PyRuntimeError::new_err(e.to_string())
}

/// Field name as written in the E57 XML, e.g. `cartesianX` or `las:pointSourceId`.
fn record_name(name: &RecordName) -> String {
    let standard = match name {
        RecordName::CartesianX => "cartesianX",
        RecordName::CartesianY => "cartesianY",
        RecordName::CartesianZ => "cartesianZ",
        RecordName::CartesianInvalidState => "cartesianInvalidState",
        RecordName::SphericalRange => "sphericalRange",
        RecordName::SphericalAzimuth => "sphericalAzimuth",
        RecordName::SphericalElevation => "sphericalElevation",
        RecordName::SphericalInvalidState => "sphericalInvalidState",
        RecordName::Intensity => "intensity",
        RecordName::IsIntensityInvalid => "isIntensityInvalid",
        RecordName::ColorRed => "colorRed",
        RecordName::ColorGreen => "colorGreen",
        RecordName::ColorBlue => "colorBlue",
        RecordName::IsColorInvalid => "isColorInvalid",
        RecordName::RowIndex => "rowIndex",
        RecordName::ColumnIndex => "columnIndex",
        RecordName::ReturnCount => "returnCount",
        RecordName::ReturnIndex => "returnIndex",
        RecordName::TimeStamp => "timeStamp",
        RecordName::IsTimeStampInvalid => "isTimeStampInvalid",
        RecordName::Unknown { namespace, name } if namespace.is_empty() => return name.clone(),
        RecordName::Unknown { namespace, name } => return format!("{namespace}:{name}"),
    };
    standard.to_string()
}

/// One output column, typed from the record's declared data type.
enum Column {
    F32(Vec<f32>),
    F64(Vec<f64>),
    U8(Vec<u8>),
    U16(Vec<u16>),
    I32(Vec<i32>),
    I64(Vec<i64>),
}

impl Column {
    fn new(data_type: &RecordDataType, capacity: usize) -> Self {
        match data_type {
            RecordDataType::Single { .. } => Column::F32(Vec::with_capacity(capacity)),
            // Scaled integers are returned as their scaled f64 values.
            RecordDataType::Double { .. } | RecordDataType::ScaledInteger { .. } => {
                Column::F64(Vec::with_capacity(capacity))
            }
            // Plain integers use the smallest type that holds the declared range.
            RecordDataType::Integer { min, max } => {
                if *min >= 0 && *max <= u8::MAX as i64 {
                    Column::U8(Vec::with_capacity(capacity))
                } else if *min >= 0 && *max <= u16::MAX as i64 {
                    Column::U16(Vec::with_capacity(capacity))
                } else if *min >= i32::MIN as i64 && *max <= i32::MAX as i64 {
                    Column::I32(Vec::with_capacity(capacity))
                } else {
                    Column::I64(Vec::with_capacity(capacity))
                }
            }
        }
    }

    fn push(&mut self, value: &RecordValue, data_type: &RecordDataType) -> PyResult<()> {
        match (self, value) {
            (Column::F32(v), RecordValue::Single(x)) => v.push(*x),
            (Column::F64(v), RecordValue::Double(x)) => v.push(*x),
            (Column::F64(v), RecordValue::ScaledInteger(_)) => {
                v.push(value.to_f64(data_type).map_err(runtime_error)?)
            }
            (Column::U8(v), RecordValue::Integer(x)) => v.push(*x as u8),
            (Column::U16(v), RecordValue::Integer(x)) => v.push(*x as u16),
            (Column::I32(v), RecordValue::Integer(x)) => v.push(*x as i32),
            (Column::I64(v), RecordValue::Integer(x)) => v.push(*x),
            _ => {
                return Err(runtime_error(format!(
                    "record value {value:?} does not match its declared type {data_type:?}"
                )))
            }
        }
        Ok(())
    }

    fn into_numpy(self, py: Python<'_>) -> Py<PyAny> {
        match self {
            Column::F32(v) => PyArray1::from_vec(py, v).into_any().unbind(),
            Column::F64(v) => PyArray1::from_vec(py, v).into_any().unbind(),
            Column::U8(v) => PyArray1::from_vec(py, v).into_any().unbind(),
            Column::U16(v) => PyArray1::from_vec(py, v).into_any().unbind(),
            Column::I32(v) => PyArray1::from_vec(py, v).into_any().unbind(),
            Column::I64(v) => PyArray1::from_vec(py, v).into_any().unbind(),
        }
    }
}

/// Reads every field of one scan exactly as stored in the file.
///
/// Returns a dict mapping E57 field names (`cartesianX`, `colorRed`,
/// `rowIndex`, `returnIndex`, `timeStamp`, extension fields such as
/// `las:pointSourceId`, ...) to 1-D numpy arrays with one entry per record.
///
/// Nothing is transformed: no pose, no spherical-to-cartesian conversion,
/// no normalisation, and invalid points are kept (use the
/// `cartesianInvalidState` / `sphericalInvalidState` fields to filter them).
/// Floats keep their stored precision, scaled integers are returned as
/// float64, and integers use the smallest numpy dtype that holds the field's
/// declared range (so 8-bit colour is uint8 and 16-bit colour is uint16).
///
/// `fields` restricts the result to the named fields, which also avoids
/// allocating memory for the others.
#[pyfunction]
#[pyo3(signature = (filepath, index, *, fields = None))]
fn read_scan_raw(
    py: Python<'_>,
    filepath: &str,
    index: isize,
    fields: Option<Vec<String>>,
) -> PyResult<Py<PyDict>> {
    let mut file = E57Reader::from_file(filepath).map_err(runtime_error)?;
    let pointclouds = file.pointclouds();
    let count = pointclouds.len();
    let resolved = if index < 0 {
        count.checked_sub(index.unsigned_abs())
    } else {
        Some(index as usize).filter(|&i| i < count)
    };
    let pc = resolved
        .and_then(|i| pointclouds.into_iter().nth(i))
        .ok_or_else(|| {
            PyIndexError::new_err(format!(
                "scan index {index} out of range for file with {count} scans"
            ))
        })?;

    let names: Vec<String> = pc.prototype.iter().map(|r| record_name(&r.name)).collect();
    if let Some(requested) = &fields {
        let missing: Vec<&String> = requested.iter().filter(|f| !names.contains(f)).collect();
        if !missing.is_empty() {
            return Err(PyValueError::new_err(format!(
                "fields {missing:?} not in scan; available fields: {names:?}"
            )));
        }
    }
    let wanted = |name: &String| fields.as_ref().is_none_or(|f| f.contains(name));

    let records = pc.records as usize;
    // One slot per prototype entry; None for fields that weren't requested.
    let mut columns: Vec<Option<Column>> = pc
        .prototype
        .iter()
        .zip(&names)
        .map(|(r, name)| wanted(name).then(|| Column::new(&r.data_type, records)))
        .collect();

    let reader = file.pointcloud_raw(&pc).map_err(runtime_error)?;
    for values in reader {
        let values = values.map_err(runtime_error)?;
        for ((column, value), record) in columns.iter_mut().zip(&values).zip(&pc.prototype) {
            if let Some(column) = column {
                column.push(value, &record.data_type)?;
            }
        }
    }

    let dict = PyDict::new(py);
    for (column, name) in columns.into_iter().zip(names) {
        if let Some(column) = column {
            dict.set_item(name, column.into_numpy(py))?;
        }
    }
    Ok(dict.unbind())
}

/// e57 pointcloud file reading.
#[pymodule]
fn e57(_py: Python<'_>, m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<E57>()?;
    m.add_function(wrap_pyfunction!(raw_xml, m)?)?;
    m.add_function(wrap_pyfunction!(read_points, m)?)?;
    m.add_function(wrap_pyfunction!(read_scan_raw, m)?)?;
    Ok(())
}
