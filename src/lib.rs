use std::fs::File;
use std::io::BufReader;

use ::e57::{CartesianCoordinate, E57Reader};
use ndarray::Ix2;
use numpy::PyArray;
use pyo3::prelude::*;

#[pyclass]
pub struct E57 {
    #[pyo3(get)]
    pub points: Py<PyArray<f64, Ix2>>,
    #[pyo3(get)]
    pub color: Py<PyArray<f32, Ix2>>,
    #[pyo3(get)]
    pub intensity: Py<PyArray<f32, Ix2>>,
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
        points: Py::from(
            PyArray::from_vec(py, point_vec)
                .reshape((nrows, 3))
                .unwrap(),
        ),
        color: Py::from(PyArray::new(py, (0, 3), false)),
        intensity: Py::from(PyArray::new(py, (0, 1), false)),
    };
    if n_colors == n_points {
        e57.color = Py::from(
            PyArray::from_vec(py, color_vec)
                .reshape((nrows, 3))
                .unwrap(),
        )
    }
    if n_intensities == n_points {
        e57.intensity = Py::from(
            PyArray::from_vec(py, intensity_vec)
                .reshape((nrows, 1))
                .unwrap(),
        )
    }
    Ok(e57)
}

/// e57 pointcloud file reading.
#[pymodule]
fn e57(_py: Python<'_>, m: &PyModule) -> PyResult<()> {
    m.add_class::<E57>()?;
    m.add_function(wrap_pyfunction!(raw_xml, m)?)?;
    m.add_function(wrap_pyfunction!(read_points, m)?)?;
    Ok(())
}
