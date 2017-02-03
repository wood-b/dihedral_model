import numpy as np


class Histogram(object):
    def __init__(self, bin_start, bin_stop, bin_num):
        self.bin_num = bin_num
        self.bin_start = bin_start
        self.bin_stop = bin_stop
        self.bin_width = (bin_stop - bin_start) / bin_num
        self.bin_edges = np.linspace(self.bin_start, self.bin_stop, self.bin_num)
        self.counts = np.zeros(len(self.bin_edges)-1, dtype=int)
        self.bins = np.array([(self.bin_edges[i] + self.bin_edges[i + 1]) / 2.
                              for i in range(len(self.bin_edges))
                              if i < len(self.bin_edges) - 1])

    def update(self, array):
        temp_counts, bins_edges = np.histogram(array, bins=self.bin_edges)
        self.counts += temp_counts
