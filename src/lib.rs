use std::fs::File;
use std::io::BufReader;

use ::e57::{CartesianCoordinate, E57Reader};
use ndarray::{Ix2};
use numpy::PyArray;
use pyo3::prelude::*;

#[pyclass]
pub struct E57 {
    #[pyo3(get)]
    pub points: Py<PyArray<f64, Ix2>>,
    #[pyo3(get)]
    pub color: Py<PyArray<f32, Ix2>>,
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
unsafe fn read_points<'py>(py: Python<'py>, filepath: &str) -> PyResult<E57> {
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
    let mut color_vec = Vec::with_capacity(pc.records as usize * 3);
    let mut vec = Vec::with_capacity(pc.records as usize * 3);
    let mut nrows = 0;
    for pointcloud in file.pointclouds() {
        let mut iter = file
            .pointcloud_simple(&pointcloud)
            .expect("this file should contain a pointcloud");
        iter.spherical_to_cartesian(true);
        iter.cartesian_to_spherical(false);
        iter.intensity_to_color(true);
        iter.apply_pose(true);

        for p in iter {
            let p = p.expect("Unable to read next point");
            if let CartesianCoordinate::Valid { x, y, z } = p.cartesian {
                vec.extend([x, y, z]);
                nrows += 1
            }
            // if let Some(intensity) = p.intensity{
            //     vec.append(intensity as f64)
            // }
            if let Some(color) = p.color {
                color_vec.extend([color.red, color.green, color.blue])
            }
        }
    }
    if color_vec.len() == vec.len() {
        Ok(E57 {
            points: Py::from(PyArray::from_vec(py, vec).reshape((nrows, 3)).unwrap()),
            color: Py::from(PyArray::from_vec(py, color_vec).reshape((nrows, 3)).unwrap()),
        })
    } else {
        Ok(E57 {
            points: Py::from(PyArray::from_vec(py, vec).reshape((nrows, 3)).unwrap()),
            color: Py::from(PyArray::new(py, (0,3), false)),
        })
    }
}

/// e57 pointcloud file reading.
#[pymodule]
fn e57(_py: Python<'_>, m: &PyModule) -> PyResult<()> {
    m.add_class::<E57>()?;
    m.add_function(wrap_pyfunction!(raw_xml, m)?)?;
    m.add_function(wrap_pyfunction!(read_points, m)?)?;
    Ok(())
}
