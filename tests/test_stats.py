import numpy as np
import unittest
from utils.stats import Stats, ArrayStats

__author__ = "Brandon Wood"


class TestStats(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.sigma = 5.0
        cls.values = np.random.normal(20.0, cls.sigma, size=10000)
        cls.array_len = 10
        cls.arrays = np.array([np.random.uniform(0.0, 5.0, size=cls.array_len) for i in range(10)])

    def test_stats(self):
        list_vals = []
        test_stat = Stats()
        var = None
        for idx, value in enumerate(self.values, start=1):
            list_vals.append(value)
            mean = np.mean(list_vals)
            var = np.var(list_vals)
            test_stat.update(float(idx), value)
            np.testing.assert_almost_equal(mean, test_stat.get_mean)
        np.testing.assert_almost_equal(var, test_stat.get_variance, decimal=2)
        np.testing.assert_almost_equal(self.sigma, test_stat.get_stdev, decimal=0)

    def test_array_stats(self):
        array_stats = ArrayStats(self.array_len)
        for idx, x_array in enumerate(self.arrays, start=1):
            array_stats.update(float(idx), x_array)
        mean = np.array([np.mean(i) for i in self.arrays.T])
        var = np.array([np.var(i) for i in self.arrays.T])
        np.testing.assert_almost_equal(mean, array_stats.get_means, decimal=2)
        np.testing.assert_almost_equal(var, array_stats.get_variances, decimal=0)

if __name__ == '__main__':
    unittest.main()

