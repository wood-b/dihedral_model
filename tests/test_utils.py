import numpy as np
import math
import json
import unittest

import

__author__ = "Brandon Wood"

class test_util_functions(unittest.TestCase):
    def test_uvec(self):
        pt1 = np.array([0.0, 0.0, 0.0])
        pt2 = np.array([0.0, 0.0, 1.0])
        self.assertEquals(1.0, utils.unit_vector(pt1, pt2))

if __name__ == '__main__':
    unittest.main()