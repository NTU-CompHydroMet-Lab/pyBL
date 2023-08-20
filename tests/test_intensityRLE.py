from pyBL.timeseries import IntensityRLE
from pyBL.dataclasses import Cell
import numpy as np
import pytest

def test_basic_instantiation():
    rle = IntensityRLE([1, 2, 3], [0, 1, 0])
    assert rle.time.tolist() == [1, 2, 3]
    assert rle.intensity.tolist() == [0, 1, 0]

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

def test_add_with_edge_case():
    rle = IntensityRLE([10, 20, 30], [5, 10, 15])
    """
    Testing different case when adding a cell to an RLE
    case1: 
        cell.startTime < rle.time[0]
        cell.endTime < rle.time[0]
    case2:
        cell.startTime < rle.time[0]
        cell.endTime = rle.time[0]
    """
    case1 = rle + Cell(1, 5, 3)  # a sample cell with specific properties
    assert case1.time.tolist() == [1, 5, 10, 20, 30]
    assert case1.intensity.tolist() == [3, 0, 5, 10, 15]
    