use std::fs::File;
use std::io::BufReader;

use ::e57::{E57Reader, Point};
use ndarray::Ix2;
use numpy::PyArray;
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
    let mut vec = Vec::with_capacity(pc.records as usize);
    let iter = file
        .pointcloud(pc)
        .expect("this file contains a pointcloud");
    for p in iter {
        let p = p.expect("Unable to read next point");
        let p = Point::from_values(p, &pc.prototype)
            .expect("failed to convert raw point to simple point");
        if let Some(c) = p.cartesian {
            if let Some(invalid) = p.cartesian_invalid {
                if invalid != 0 {
                    continue;
                }
            }

            vec.push(vec![c.x, c.y, c.z]);
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
            vec.push(vec![x, y, z]);
        }
    }
    let pyarray = PyArray::from_vec2(py, &vec).unwrap();
    Ok(pyarray)
}

/// e57 pointcloud file reading.
#[pymodule]
fn e57(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(raw_xml, m)?)?;
    m.add_function(wrap_pyfunction!(read_points, m)?)?;
    Ok(())
}
