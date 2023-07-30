import numpy as np
import pytest

import e57


def test_raw_xml():
    raw_xml = e57.raw_xml(r"testdata/bunnyFloat.e57")
    assert "<?xml version" in raw_xml


def test_read_points():
    pointcloud = e57.read_points(r"testdata/bunnyFloat.e57")
    assert isinstance(pointcloud, np.ndarray)
    assert len(pointcloud) == 30_571


def test_box_dimensions():
    pointcloud: np.ndarray = e57.read_points(r"testdata/bunnyFloat.e57")
    max_coords = pointcloud.max(0, None, False, -np.inf)
    min_coords = pointcloud.min(0, None, False, np.inf)
    X, Y, Z = max_coords - min_coords
    assert X == pytest.approx(0.155698)
    assert Y == pytest.approx(0.14731)
    assert Z == pytest.approx(0.120672)


def test_global_box_center():
    pointcloud: np.ndarray = e57.read_points(r"testdata/bunnyFloat.e57")
    max_coords = pointcloud.max(0, None, False, -np.inf)
    min_coords = pointcloud.min(0, None, False, np.inf)
    X, Y, Z = (max_coords + min_coords) / 2
    assert X == pytest.approx(-0.016840)
    assert Y == pytest.approx(0.113666)
    assert Z == pytest.approx(-0.001537)
