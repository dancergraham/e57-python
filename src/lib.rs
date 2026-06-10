use std::fs::File;
use std::io::BufReader;

use ::e57::{CartesianCoordinate, E57Reader};
use numpy::ndarray::Array2;
use numpy::{IntoPyArray, PyArray2};
use pyo3::prelude::*;

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

/// e57 pointcloud file reading.
#[pymodule]
fn e57(_py: Python<'_>, m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<E57>()?;
    m.add_function(wrap_pyfunction!(raw_xml, m)?)?;
    m.add_function(wrap_pyfunction!(read_points, m)?)?;
    Ok(())
}
