from pyBL.timeseries import IntensityRLE
from pyBL.dataclasses import Cell
import numpy as np
import pytest


@pytest.mark.parametrize(
    "time, intensity, expected_time, expected_intensity",
    [
        ([1, 2, 3], [0, 1, 0], [1., 2., 3.], [0., 1., 0.]),
    ]
)
def test_basic_instantiation(time, intensity, expected_time, expected_intensity):
    rle = IntensityRLE(time, intensity)
    assert rle.time.tolist() == expected_time
    assert rle.intensity.tolist() == expected_intensity

@pytest.mark.parametrize(
    "time, intensity, error_msg",
    [
        ([10, 20, 30], None, "time and intensity must both be None or not None"),
        (None, [3, 6, 9], "time and intensity must both be None or not None"),
        ([10, 20, 30], [3, 6], "time and intensity must have the same length"),
        ([10, 20], [3, 6, 9], "time and intensity must have the same length"),
        ([10, 20, 30, 40], np.array([[3, 6], [9, 12]]), "time and intensity must be 1D arrays"),
        ([10, 20, 30, 40], np.array([[3], [6], [9], [12]]), "time and intensity must be 1D arrays"),
        ([10, 10, 9, 10], [3, 6, 9, 12], "time must be monotonically increasing"),
        ([10, 20, np.inf, 40], [3, 6, 9, 12], "time must be monotonically increasing"),
        ([10, 20, 40, np.inf], [3, 6, 9, 12], "Last intensity must be 0"),
    ]
)
def test_basic_instatiation_fail(time, intensity, error_msg):
    with pytest.raises(ValueError, match=error_msg):
        IntensityRLE(time, intensity)

def test_mismatched_time_intensity():
    with pytest.raises(ValueError, match="time and intensity must have the same length"):
        IntensityRLE([1, 2], [0, 1, 0])

def test_from_cells():
    # assuming Cell is another class you've defined
    cell1 = Cell(5, 13, 7)  
    cell2 = Cell(8, 21, 3)
    rle = IntensityRLE.fromCells([cell1, cell2])
    assert rle.time.tolist() == [5, 8, 13, 21]
    assert rle.intensity.tolist() == [7, 10, 3, 0]


@pytest.mark.parametrize(
    "new_cell, expected_time, expected_intensity",
    [   (Cell(1, 5, 3), [1., 5., 10., 20., 30., 40.], [3., 0., 5., 10., 15., 0.]),
        (Cell(1, 10, 3), [1., 10., 20., 30., 40], [3., 5., 10., 15., 0.]),
        (Cell(1, 15, 3), [1., 10., 15., 20., 30., 40], [3., 8., 5., 10., 15., 0.]),
        (Cell(1, 20, 3), [1., 10., 20., 30., 40], [3., 8., 10., 15., 0.]),
        (Cell(1, 40, 3), [1., 10., 20., 30., 40], [3., 8., 13., 18., 0.]),
        (Cell(1, 41, 3), [1., 10., 20., 30., 40, 41], [3., 8., 13., 18., 3., 0.]),
        (Cell(-100, 25, 3), [-100., 10., 20., 25., 30., 40], [3., 8., 13., 10., 15., 0.]),
    ]
)
def test_operational_add(new_cell, expected_time, expected_intensity):
    rle = IntensityRLE([10, 20, 30, 40], [5, 10, 15, 0])
    """
    Testing different case when adding a cell to an RLE
    case1: 
        cell.startTime < rle.time[0]
        cell.endTime < rle.time[0]
    case2:
        cell.startTime < rle.time[0]
        cell.endTime = rle.time[0]
    """
    result = rle + new_cell  # a sample cell with specific properties
    assert result.time.tolist() == expected_time
    assert result.intensity.tolist() == expected_intensity



@pytest.mark.parametrize(
    "new_cell, times, intensities",
    [   (Cell(1, 5, 3), [-5, 1, 3, 5, 7], [0., 3., 3., 0., 0.]),
        (Cell(1, 10, 3), [-5, 1, 9, 10, 11], [0., 3., 3., 5., 5.]),
        (Cell(1, 15, 3), [-5, 1, 10, 14, 20], [0., 3., 8., 8., 10.]),
        (Cell(1, 20, 3), [0, 1, 10, 19, 20], [0., 3., 8., 8., 10.]),
        (Cell(1, 40, 3), [-9999, 1, 39, 40, 41], [0., 3., 18., 0., 0.]),
    ]
)
def test_add(new_cell, times, intensities):
    rle = IntensityRLE([10, 20, 30, 40], [5, 10, 15, 0])

    rle.add(new_cell)
    for time, intensity in zip(times, intensities):
        assert rle[time] == intensity
