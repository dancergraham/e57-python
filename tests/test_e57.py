import e57
import numpy as np
import pytest


def test_raw_xml():
    raw_xml = e57.raw_xml(r"testdata/bunnyFloat.e57")
    assert "<?xml version" in raw_xml


def test_read_points():
    pointcloud = e57.read_points(r"testdata/bunnyFloat.e57")
    points = pointcloud.points
    assert isinstance(points, np.ndarray)
    assert len(points) == 30_571


def test_read_spherical():
    pointcloud = e57.read_points(r"testdata/pipeSpherical.e57")
    points = pointcloud.points
    assert isinstance(points, np.ndarray)
    assert len(points) == 1_220


def test_read_color():
    pointcloud = e57.read_points(r"testdata/pipeSpherical.e57")
    color = pointcloud.color
    assert isinstance(color, np.ndarray)
    assert len(color) == 1_220


def test_read_intensity():
    pointcloud = e57.read_points(r"testdata/pipeSpherical.e57")
    intensity = pointcloud.intensity
    assert isinstance(intensity, np.ndarray)
    assert len(intensity) == 1_220
    assert np.all(intensity >= 0.3935)
    assert np.all(intensity <= 0.5555)


def test_no_rgb_intensity():
    pointcloud = e57.read_points(r"testdata/bunnyFloat.e57")
    intensity = pointcloud.intensity
    assert isinstance(intensity, np.ndarray)
    assert len(intensity) == 0


def test_box_dimensions():
    pointcloud: np.ndarray = e57.read_points(r"testdata/bunnyFloat.e57")
    points = pointcloud.points
    max_coords = points.max(0, None, False, -np.inf)
    min_coords = points.min(0, None, False, np.inf)
    X, Y, Z = max_coords - min_coords
    assert X == pytest.approx(0.155698)
    assert Y == pytest.approx(0.14731)
    assert Z == pytest.approx(0.120672)


def test_global_box_center():
    pointcloud: np.ndarray = e57.read_points(r"testdata/bunnyFloat.e57")
    max_coords = pointcloud.points.max(0, None, False, -np.inf)
    min_coords = pointcloud.points.min(0, None, False, np.inf)
    X, Y, Z = (max_coords + min_coords) / 2
    assert X == pytest.approx(-0.016840)
    assert Y == pytest.approx(0.113666)
    assert Z == pytest.approx(-0.001537)


def test_file_not_found():
    raised = False
    try:
        e57.read_points(r"testdata/filenotfound.e57")
    except RuntimeError as e:
        raised = True
        assert "Failed to read E57" in str(e)
        assert "Unable to open file" in str(e)
    assert raised


def test_empty_file():
    raised = False
    try:
        e57.read_points(r"testdata/empty.e57")
    except RuntimeError as e:
        raised = True
        assert "Failed to read E57" in str(e)
        assert "Failed to read E57 file header" in str(e)
    assert raised


def test_invalid_file():
    raised = False
    try:
        e57.read_points(r"testdata/invalid.e57")
    except RuntimeError as e:
        raised = True
        assert "Failed to read E57" in str(e)
        assert "Failed to read E57 file header" in str(e)
    assert raised


def test_just_xml():
    raised = False
    try:
        e57.read_points(r"testdata/justxml.e57")
    except RuntimeError as e:
        raised = True
        assert "Invalid E57 content" in str(e)
        assert "Found unsupported signature in header" in str(e)
    assert raised


def test_raw_xml_file_not_found():
    raised = False
    try:
        e57.raw_xml(r"testdata/filenotfound.e57")
    except FileNotFoundError:
        raised = True
    assert raised


def test_raw_xml_empty():
    raised = False
    try:
        e57.raw_xml(r"testdata/empty.e57")
    except RuntimeError as e:
        raised = True
        assert "Failed to read E57" in str(e)
        assert "Cannot read page size bytes" in str(e)
    assert raised


def test_raw_xml_invalid():
    raised = False
    try:
        e57.raw_xml(r"testdata/invalid.e57")
    except RuntimeError as e:
        raised = True
        assert "Failed to read E57" in str(e)
        assert "Cannot read page size bytes" in str(e)
    assert raised


def test_raw_xml_just_xml():
    raised = False
    try:
        e57.raw_xml(r"testdata/justxml.e57")
    except RuntimeError as e:
        raised = True
        assert "Failed to read E57" in str(e)
        assert "Failed creating paged CRC reader" in str(e)
    assert raised
