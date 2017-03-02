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
        std_error = None
        for idx, value in enumerate(self.values, start=1):
            list_vals.append(value)
            mean = np.mean(list_vals)
            var = np.var(list_vals)
            std_error = np.sqrt(var) / np.sqrt(idx)
            test_stat.update(value)
            np.testing.assert_allclose(test_stat.mean, mean)
        np.testing.assert_allclose(test_stat.variance, var, rtol=1e-2)
        np.testing.assert_allclose(test_stat.std_error, std_error, rtol=1e-3)
        np.testing.assert_allclose(test_stat.stdev, self.sigma, rtol=1e0)

    def test_array_stats(self):
        array_stats = ArrayStats(self.array_len)
        for idx, x_array in enumerate(self.arrays, start=1):
            array_stats.update(x_array)
        mean = np.array([np.mean(i) for i in self.arrays.T])
        var = np.array([np.var(i) for i in self.arrays.T])
        stdev = np.sqrt(var)
        std_error = stdev / np.sqrt(10.)
        np.testing.assert_allclose(array_stats.mean, mean, rtol=1e-2)
        np.testing.assert_allclose(array_stats.variance, var, rtol=1e0)
        np.testing.assert_allclose(array_stats.stdev, stdev, rtol=1e0)
        np.testing.assert_allclose(array_stats.std_error, std_error, rtol=1e-1)


if __name__ == '__main__':
    unittest.main()
