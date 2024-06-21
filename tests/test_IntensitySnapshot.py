from pybl.timeseries import IndexedSnapshot
import numpy as np
import pytest


@pytest.mark.parametrize(
    "time, intensity, error_msg",
    [
        ([10, 20, 30], None, "time and intensity must both be None or not None"),
        (None, [3, 6, 9], "time and intensity must both be None or not None"),
        ([10, 20, 30], [3, 6], "time and intensity must have the same length"),
        ([10, 20], [3, 6, 9], "time and intensity must have the same length"),
        (
            [10, 20, 30, 40],
            np.array([[3, 6], [9, 12]]),
            "time and intensity must be 1D arrays",
        ),
        (
            [10, 20, 30, 40],
            np.array([[3], [6], [9], [12]]),
            "time and intensity must be 1D arrays",
        ),
        ([10, 10, 9, 10], [3, 6, 9, 12], "time must be strictly increasing"),
        ([10, 20, 9999999, 40], [3, 6, 9, 12], "time must be strictly increasing"),
    ],
)
def test_basic_instatiation_fail(time, intensity, error_msg):
    with pytest.raises(ValueError, match=error_msg):
        IndexedSnapshot(time, intensity)


def test_mismatched_time_intensity():
    with pytest.raises(
        ValueError, match="time and intensity must have the same length"
    ):
        IndexedSnapshot([1, 2], [0, 1, 0])


@pytest.mark.parametrize(
    "time, intensity, expected_time, expected_intensity",
    [
        ([-10, 0, 10, 20], [1, 2, 3, 4], [-10, 0, 10, 20, 21], [1, 2, 3, 4, np.nan]),
        ([-10, 0, 10, 20], [1, 2, 2, 4], [-10, 0, 20, 21], [1, 2, 4, np.nan]),
        ([-10, 0, 10, 20], [1, 2, 3, 3], [-10, 0, 10, 21], [1, 2, 3, np.nan]),
        ([-10, 0, 10, 20], [1, 2, 2, 2], [-10, 0, 21], [1, 2, np.nan]),
        ([-10, 0, 10, 20], [1, 1, 1, 1], [-10, 21], [1, np.nan]),
        ([-10, 0, 10, 20], [1, 1, 1, 0], [-10, 20, 21], [1, 0, np.nan]),
        ([-10, 0, 10, 20], [1, 2, 3, np.nan], [-10, 0, 10, 20], [1, 2, 3, np.nan]),
        ([-10, 0, 10, 20], [1, 2, np.nan, np.nan], [-10, 0, 10], [1, 2, np.nan]),
    ],
)
def test_basic_instantiation(time, intensity, expected_time, expected_intensity):
    ist = IndexedSnapshot(time, intensity)
    assert np.array_equal(ist.time, np.array(expected_time))
    assert np.array_equal(ist.intensity, expected_intensity, equal_nan=True)


@pytest.mark.parametrize(
    "time, intensity, expected_time, expected_intensity",
    [
        (
            [0.123, 0.456, 0.789],
            [0.789, 0.456, 0.123],
            [0.123, 0.456, 0.789, 1.789],
            [0.789, 0.456, 0.123, np.nan],
        ),
    ],
)
def test_float_instantiation(time, intensity, expected_time, expected_intensity):
    ist = IndexedSnapshot(time, intensity)
    assert np.allclose(ist.time, expected_time)
    assert np.allclose(ist.intensity, expected_intensity, equal_nan=True)


# def test_from_cells():
#    # assuming Cell is another class you've defined
#    cell1 = ConstantCell(5, 13, 7)
#    cell2 = ConstantCell(8, 21, 3)
#    rle = IndexedSnapshot.fromCells([cell1, cell2])
#    assert rle.time.tolist() == [5, 8, 13, 21, 22]
#    assert np.array_equal(rle.intensity.tolist(), [7, 10, 3, 0, np.nan], equal_nan=True)


# @pytest.mark.parametrize(
#    "new_cell, expected_time, expected_intensity",
#    [
#        (
#            ConstantCell(1, 5, 3),
#            [1.0, 5.0, 10.0, 20.0, 30.0, 40.0, 41.0],
#            [3.0, 0.0, 5.0, 10.0, 15.0, 0.0, np.nan],
#        ),
#        (
#            ConstantCell(1, 10, 3),
#            [1.0, 10.0, 20.0, 30.0, 40.0, 41.0],
#            [3.0, 5.0, 10.0, 15.0, 0.0, np.nan],
#        ),
#        (
#            ConstantCell(1, 15, 3),
#            [1.0, 10.0, 15.0, 20.0, 30.0, 40.0, 41.0],
#            [3.0, 8.0, 5.0, 10.0, 15.0, 0.0, np.nan],
#        ),
#        (
#            ConstantCell(1, 20, 3),
#            [1.0, 10.0, 20.0, 30.0, 40.0, 41.0],
#            [3.0, 8.0, 10.0, 15.0, 0.0, np.nan],
#        ),
#        (
#            ConstantCell(1, 40, 3),
#            [1.0, 10.0, 20.0, 30.0, 40.0, 41.0],
#            [3.0, 8.0, 13.0, 18.0, 0.0, np.nan],
#        ),
#        (
#            ConstantCell(1, 41, 3),
#            [1.0, 10.0, 20.0, 30.0, 40.0, 41.0, 42.0],
#            [3.0, 8.0, 13.0, 18.0, 3.0, 0.0, np.nan],
#        ),
#        (
#            ConstantCell(-100, 25, 3),
#            [-100.0, 10.0, 20.0, 25.0, 30.0, 40.0, 41.0],
#            [3.0, 8.0, 13.0, 10.0, 15.0, 0.0, np.nan],
#        ),
#    ],
# )
# def test_operational_add(new_cell, expected_time, expected_intensity):
#    rle = IndexedSnapshot([10, 20, 30, 40], [5, 10, 15, 0])
#    """
#    Testing different case when adding a cell to an RLE
#    case1:
#        cell.startTime < rle.time[0]
#        cell.endTime < rle.time[0]
#    case2:
#        cell.startTime < rle.time[0]
#        cell.endTime = rle.time[0]
#    """
#    result = rle + new_cell  # a sample cell with specific properties
#    assert result.time.tolist() == expected_time
#    assert np.array_equal(result.intensity.tolist(), expected_intensity, equal_nan=True)


# @pytest.mark.parametrize(
#    "new_cell, times, intensities",
#    [
#        (ConstantCell(1, 5, 3), [-5, 1, 3, 5, 7], [0.0, 3.0, 3.0, 0.0, 0.0]),
#        (ConstantCell(1, 10, 3), [-5, 1, 9, 10, 11], [0.0, 3.0, 3.0, 5.0, 5.0]),
#        (ConstantCell(1, 15, 3), [-5, 1, 10, 14, 20], [0.0, 3.0, 8.0, 8.0, 10.0]),
#        (ConstantCell(1, 20, 3), [0, 1, 10, 19, 20], [0.0, 3.0, 8.0, 8.0, 10.0]),
#        (ConstantCell(1, 40, 3), [-9999, 1, 39, 40, 41], [0.0, 3.0, 18.0, 0.0, 0.0]),
#    ],
# )
# def test_add(new_cell, times, intensities):
#    rle = IndexedSnapshot([10, 20, 30, 40], [5, 10, 15, 0])

#    rle.add(new_cell)
#    for time, intensity in zip(times, intensities):
#        assert rle[time] == intensity
