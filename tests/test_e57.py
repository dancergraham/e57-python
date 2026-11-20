import e57
import numpy as np
import pytest

BUNNY_N_POINTS = 30_571
PIPE_N_POINTS = 1_220


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


# ---------------------------------------------------------------------------
# Shape assertions
# ---------------------------------------------------------------------------


def test_read_points_shape():
    pointcloud = e57.read_points(r"testdata/bunnyFloat.e57")
    assert pointcloud.points.shape == (BUNNY_N_POINTS, 3)


def test_read_spherical_shape():
    pointcloud = e57.read_points(r"testdata/pipeSpherical.e57")
    assert pointcloud.points.shape == (PIPE_N_POINTS, 3)


def test_read_color_shape():
    pointcloud = e57.read_points(r"testdata/pipeSpherical.e57")
    assert pointcloud.color.shape == (PIPE_N_POINTS, 3)


def test_read_intensity_shape():
    pointcloud = e57.read_points(r"testdata/pipeSpherical.e57")
    assert pointcloud.intensity.shape == (PIPE_N_POINTS, 1)


# ---------------------------------------------------------------------------
# Dtype assertions
# ---------------------------------------------------------------------------


def test_read_points_dtype():
    pointcloud = e57.read_points(r"testdata/bunnyFloat.e57")
    assert pointcloud.points.dtype == np.float64


def test_read_color_dtype():
    pointcloud = e57.read_points(r"testdata/pipeSpherical.e57")
    assert pointcloud.color.dtype == np.float32


def test_read_intensity_dtype():
    pointcloud = e57.read_points(r"testdata/pipeSpherical.e57")
    assert pointcloud.intensity.dtype == np.float32


# ---------------------------------------------------------------------------
# Partial-attribute shape tests (file has no color or intensity)
# ---------------------------------------------------------------------------


def test_no_color_shape():
    """When a file has no color data the returned array should be (0, 3)."""
    pointcloud = e57.read_points(r"testdata/bunnyFloat.e57")
    assert pointcloud.color.shape == (0, 3)
    assert pointcloud.color.dtype == np.float32


def test_no_intensity_shape():
    """When a file has no intensity data the returned array should be (0, 1)."""
    pointcloud = e57.read_points(r"testdata/bunnyFloat.e57")
    assert pointcloud.intensity.shape == (0, 1)
    assert pointcloud.intensity.dtype == np.float32


# ---------------------------------------------------------------------------
# Consistency between arrays
# ---------------------------------------------------------------------------


def test_points_color_intensity_row_consistency():
    """points, color, and intensity must share the same row count."""
    pointcloud = e57.read_points(r"testdata/pipeSpherical.e57")
    n = pointcloud.points.shape[0]
    assert pointcloud.color.shape[0] == n
    assert pointcloud.intensity.shape[0] == n


def test_partial_attribute_row_consistency():
    """When color/intensity are absent their row count should be 0, not equal to points."""
    pointcloud = e57.read_points(r"testdata/bunnyFloat.e57")
    assert pointcloud.points.shape[0] > 0
    assert pointcloud.color.shape[0] == 0
    assert pointcloud.intensity.shape[0] == 0


# ---------------------------------------------------------------------------
# Value-range checks
# ---------------------------------------------------------------------------


def test_color_values_normalized():
    """Float color values should lie in [0.0, 1.0]."""
    pointcloud = e57.read_points(r"testdata/pipeSpherical.e57")
    assert np.all(pointcloud.color >= 0.0)
    assert np.all(pointcloud.color <= 1.0)


# ---------------------------------------------------------------------------
# Spherical-to-Cartesian conversion
# ---------------------------------------------------------------------------


def test_spherical_conversion_finite():
    """Spherical data converted to Cartesian should contain only finite values."""
    pointcloud = e57.read_points(r"testdata/pipeSpherical.e57")
    assert np.all(np.isfinite(pointcloud.points))


def test_spherical_conversion_three_columns():
    """Spherical data converted to Cartesian should produce an (n, 3) array."""
    pointcloud = e57.read_points(r"testdata/pipeSpherical.e57")
    assert pointcloud.points.ndim == 2
    assert pointcloud.points.shape[1] == 3


# ---------------------------------------------------------------------------
# raw_xml structure
# ---------------------------------------------------------------------------


def test_raw_xml_structure():
    """raw_xml should return a valid XML document containing the e57Root element."""
    xml = e57.raw_xml(r"testdata/bunnyFloat.e57")
    assert xml.startswith("<?xml")
    assert "e57Root" in xml


# ---------------------------------------------------------------------------
# Error-path tests using pytest.raises
# ---------------------------------------------------------------------------


def test_file_not_found_raises():
    with pytest.raises(RuntimeError, match="Failed to read E57"):
        e57.read_points(r"testdata/filenotfound.e57")


def test_empty_file_raises():
    with pytest.raises(RuntimeError, match="Failed to read E57 file header"):
        e57.read_points(r"testdata/empty.e57")


def test_invalid_file_raises():
    with pytest.raises(RuntimeError, match="Failed to read E57 file header"):
        e57.read_points(r"testdata/invalid.e57")


def test_just_xml_raises():
    with pytest.raises(RuntimeError, match="Found unsupported signature in header"):
        e57.read_points(r"testdata/justxml.e57")


def test_raw_xml_file_not_found_raises():
    with pytest.raises(FileNotFoundError):
        e57.raw_xml(r"testdata/filenotfound.e57")


def test_raw_xml_empty_raises():
    with pytest.raises(RuntimeError, match="Cannot read page size bytes"):
        e57.raw_xml(r"testdata/empty.e57")
