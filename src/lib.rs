use ::e57::E57Reader;
use pyo3::prelude::*;
use std::fs::File;
use std::io::BufReader;

/// Extracts the xml contents from an e57 file.
#[pyfunction]
fn raw_xml(filepath: &str) -> PyResult<String> {
    let file = File::open(filepath)?;
    let reader = BufReader::new(file);
    let xml = E57Reader::raw_xml(reader);
    let xml = match &xml {
        Ok(_) => xml,
        Err(e) => return Err(PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string())),
    };
    let xml_string = String::from_utf8(xml.unwrap())?;
    Ok(xml_string)
}

/// e57 pointcloud file reading.
#[pymodule]
fn e57(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(raw_xml, m)?)?;
    Ok(())
}
