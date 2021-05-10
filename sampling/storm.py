class Storm():
    def __init__(self, nCells):
        self.sDT = 0
        self.eDT = 0
        self.nCells = nCells
        self.cells = [None] * nCells


class Cell():
    def __init__(self):
        self.sDT = 0
        self.eDT = 0
        self.Depth = 0