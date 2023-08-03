use std::fs::File;
use std::io::BufReader;

use ::e57::{E57Reader, Point};
use ndarray::{array, Array2, Ix2};
use numpy::{IntoPyArray, PyArray};
use pyo3::prelude::*;

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
fn read_points<'py>(py: Python<'py>, filepath: &str) -> PyResult<&'py PyArray<f64, Ix2>> {
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
    let mut arr = Array2::zeros((pc.records as usize, 3));

    let iter = file
        .pointcloud(pc)
        .expect("this file contains a pointcloud");
    for (i, p) in iter.enumerate() {
        let p = p.expect("Unable to read next point");
        let p = Point::from_values(p, &pc.prototype)
            .expect("failed to convert raw point to simple point");
        let mut row = arr.row_mut(i);
        if let Some(c) = p.cartesian {
            if let Some(invalid) = p.cartesian_invalid {
                            if invalid != 0 {
                    continue;
                }
                        }

            let coordinates = array![c.x, c.y, c.z];
            row.assign(&coordinates);
        } else if let Some(s) = p.spherical {
            if let Some(invalid) = p.spherical_invalid {
                if invalid != 0 {
                    continue;
                }
            }
            let cos_ele = f64::cos(s.elevation);
            let x = s.range * cos_ele * f64::cos(s.azimuth);
            let y = s.range * cos_ele * f64::sin(s.azimuth);
            let z = s.range * f64::sin(s.elevation);
            let coordinates = array![x, y, z];
            row.assign(&coordinates);
        }
    }

    Ok(arr.into_pyarray(py))
}

/// e57 pointcloud file reading.
#[pymodule]
fn e57(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(raw_xml, m)?)?;
    m.add_function(wrap_pyfunction!(read_points, m)?)?;
    Ok(())
}
