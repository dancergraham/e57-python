import e57
import numpy as np
import pytest


def test_scan_count():
    assert e57.scan_count(r"testdata/bunnyFloat.e57") == 1
    assert e57.scan_count(r"testdata/invalidPoints.e57") == 2


def test_scan_count_file_not_found():
    with pytest.raises(RuntimeError, match="Unable to open file"):
        e57.scan_count(r"testdata/filenotfound.e57")


def test_read_scan_matches_read_points_single_scan():
    whole = e57.read_points(r"testdata/pipeSpherical.e57")
    scan = e57.read_scan(r"testdata/pipeSpherical.e57", 0)
    np.testing.assert_array_equal(scan.points, whole.points)
    np.testing.assert_array_equal(scan.color, whole.color)
    np.testing.assert_array_equal(scan.intensity, whole.intensity)


def test_read_scan_each_scan():
    # testdata/invalidPoints.e57: scan 0 has 30 points, scan 1 has 60; every third is invalid
    first = e57.read_scan(r"testdata/invalidPoints.e57", 0)
    second = e57.read_scan(r"testdata/invalidPoints.e57", 1)
    assert first.points.shape == (20, 3)
    assert second.points.shape == (40, 3)
    assert second.color.shape == (40, 3)
    assert second.intensity.shape == (40, 1)
    # scan 1, point 1 is (101, 2, -1) locally; its pose rotates 90 degrees about z
    # and translates by (10, 0, 0)
    np.testing.assert_allclose(second.points[0], [8.0, 101.0, -1.0], atol=1e-9)


def test_read_scan_without_transform():
    local = e57.read_scan(r"testdata/invalidPoints.e57", 1, transform=False)
    np.testing.assert_allclose(local.points[0], [101.0, 2.0, -1.0], atol=1e-9)


def test_read_scan_negative_index():
    last = e57.read_scan(r"testdata/invalidPoints.e57", -1)
    second = e57.read_scan(r"testdata/invalidPoints.e57", 1)
    np.testing.assert_array_equal(last.points, second.points)


@pytest.mark.parametrize("index", [2, -3])
def test_read_scan_index_out_of_range(index):
    with pytest.raises(IndexError, match="out of range"):
        e57.read_scan(r"testdata/invalidPoints.e57", index)
