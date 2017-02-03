import numpy as np
import unittest
from utils.histogram import Histogram

__author__ = "Brandon Wood"


class TestStats(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.values = np.random.uniform(-1.0, 1.0, size=10000)
        cls.values_1 = cls.values[:5000]
        cls.values_2 = cls.values[5000:]
        cls.bin_edges = np.linspace(-1.0, 1.0, 31)

    def test_histogram(self):
        counts, bins_edges = np.histogram(self.values, bins=self.bin_edges)
        test_hist = Histogram(-1.0, 1.0, 31)
        test_hist.update(self.values_1)
        test_hist.update(self.values_2)
        test_counts = test_hist.counts
        test_bin_edges = test_hist.bin_edges
        np.testing.assert_equal(counts, test_counts)
        np.testing.assert_almost_equal(bins_edges, test_bin_edges)

if __name__ == '__main__':
    unittest.main()
