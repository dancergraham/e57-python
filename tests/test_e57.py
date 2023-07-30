import e57
import numpy as np


def test_raw_xml():
    raw_xml = e57.raw_xml(r"testdata/bunnyFloat.e57")
    assert "<?xml version" in raw_xml


def test_read_points():
    pointcloud = e57.read_points(r"testdata/bunnyFloat.e57")
    assert isinstance(pointcloud, np.ndarray)
    assert len(pointcloud) == 30_571
