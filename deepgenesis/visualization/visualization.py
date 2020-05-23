from matplotlib import pyplot as plt
from dataset.dataset import DataSet

class Visualization(DataSet):

    def __init__(self, rows=2, cols=8):
        super(Visualization, self).__init__(batch_size=(rows*cols))
        self.train, self.test = self.load_trainset(), self.load_testset()
        self.rows = rows
        self.cols = cols