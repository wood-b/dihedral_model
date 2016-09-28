import numpy as np
import unittest
from utils.stats import Stats

__author__ = "Brandon Wood"


class TestStats(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.sigma = 5.0
        cls.values = np.random.normal(20.0, cls.sigma, size=10000)

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
        np.testing.assert_almost_equal(self.sigma, test_stat.get_stdev, decimal=1)

if __name__ == '__main__':
    unittest.main()
