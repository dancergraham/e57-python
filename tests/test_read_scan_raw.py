import e57
import numpy as np
import pytest

GRID = r"testdata/gridFields.e57"  # see testdata/generate_grid_fields.py


def test_all_fields_returned():
    raw = e57.read_scan_raw(GRID, 0)
    assert list(raw) == [
        "cartesianX", "cartesianY", "cartesianZ", "intensity",
        "colorRed", "colorGreen", "colorBlue",
        "rowIndex", "columnIndex", "cartesianInvalidState",
    ]
    assert all(isinstance(v, np.ndarray) and v.shape == (12,) for v in raw.values())


def test_grid_indices():
    raw = e57.read_scan_raw(GRID, 0)
    np.testing.assert_array_equal(raw["rowIndex"], np.arange(12) // 4)
    np.testing.assert_array_equal(raw["columnIndex"], np.arange(12) % 4)


def test_invalid_points_kept_with_state():
    raw = e57.read_scan_raw(GRID, 0)
    np.testing.assert_array_equal(raw["cartesianInvalidState"], np.where(np.arange(12) % 3 == 0, 2, 0))
    # read_points drops the invalid ones
    assert len(e57.read_points(GRID).points) == 8


def test_pose_not_applied():
    # The scan's pose translates by (100, 0, 0); raw values are in the local frame.
    raw = e57.read_scan_raw(GRID, 0)
    np.testing.assert_allclose(raw["cartesianX"], np.arange(12))
    assert e57.read_points(GRID).points[:, 0].min() >= 100


def test_color_not_normalised():
    raw = e57.read_scan_raw(GRID, 0)
    assert raw["colorRed"].dtype == np.uint8
    np.testing.assert_array_equal(raw["colorRed"], np.arange(12) * 20)


def test_16_bit_color_and_returns():
    # LAS-derived file with 16-bit colour, multiple returns and an extension field
    raw = e57.read_scan_raw(r"testdata/returnIndex.e57", 0)
    assert raw["colorRed"].dtype == np.uint16
    assert raw["colorRed"].max() > 255
    # Stored as converted from LAS (1-based), returned as-is
    np.testing.assert_array_equal(raw["returnIndex"], 1)
    np.testing.assert_array_equal(raw["returnCount"], 1)
    assert "las:pointSourceId" in raw
    assert len(raw["cartesianX"]) == 153


def test_scaled_integer_returned_as_float64():
    raw = e57.read_scan_raw(r"testdata/scaledIntegerIntensity.e57", 0)
    assert raw["intensity"].dtype == np.float64
    np.testing.assert_allclose(raw["intensity"], [-66.6, 66.6])


def test_field_subset():
    raw = e57.read_scan_raw(GRID, 0, fields=["rowIndex", "cartesianX"])
    assert sorted(raw) == ["cartesianX", "rowIndex"]


def test_unknown_field():
    with pytest.raises(ValueError, match="not in scan"):
        e57.read_scan_raw(GRID, 0, fields=["timeStamp"])


@pytest.mark.parametrize("index", [1, -2])
def test_index_out_of_range(index):
    with pytest.raises(IndexError, match="out of range"):
        e57.read_scan_raw(GRID, index)


def test_negative_index():
    np.testing.assert_array_equal(
        e57.read_scan_raw(GRID, -1)["cartesianX"], e57.read_scan_raw(GRID, 0)["cartesianX"]
    )


def test_file_not_found():
    with pytest.raises(RuntimeError, match="Unable to open file"):
        e57.read_scan_raw(r"testdata/filenotfound.e57", 0)
