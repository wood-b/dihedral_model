import numpy as np
#import math
#import json
import unittest

from utils import utils

__author__ = "Brandon Wood"


class TestUtilFunctions(unittest.TestCase):
    @staticmethod
    def test_uvec():
        pt1 = np.array([0.0, 0.0, 0.0])
        pt2 = np.array([1.0, 0.0, 0.0])
        uvec = np.array([1.0, 0.0, 0.0])
        np.testing.assert_array_equal(uvec, utils.unit_vector(pt1, pt2))

    def test_point_rotate(self):
        # rotate pt on x-axis around z-axis
        pt = np.array([2.0, 0.0, 0.0])
        uvec = np.array([0.0, 0.0, 1.0])
        pt_n90 = np.array([0.0, 2.0, 0.0])
        pt_0 = np.array([-2.0, 0.0, 0.0])
        pt_90 = np.array([0.0, -2.0, 0.0])
        pt_180 = pt
        pt_n180 = pt
        np.testing.assert_almost_equal(utils.point_rotation(pt, -90, uvec), pt_n90)
        np.testing.assert_almost_equal(utils.point_rotation(pt, -0, uvec), pt_0)
        np.testing.assert_almost_equal(utils.point_rotation(pt, 90, uvec), pt_90)
        np.testing.assert_almost_equal(utils.point_rotation(pt, 180, uvec), pt_180)
        np.testing.assert_almost_equal(utils.point_rotation(pt, -180, uvec), pt_n180)

    def test_eV_to_kJmol(self):
        value = 2 * 96.48533646
        test_val = utils.eV_to_kJmol([2])
        np.testing.assert_almost_equal(value, test_val)

    def test_eV_to_kcalmol(self):
        value = 2 * 23.06054887
        test_val = utils.eV_to_kcalmol([2])
        np.testing.assert_almost_equal(value, test_val)

    def test_rel_energy(self):
        energy = [2.0, 4.0, 7.0, 5.0]
        value = [0.0, 2.0, 5.0, 3.0]
        test_val = utils.relative_energy(energy)
        np.testing.assert_array_equal(value, test_val)

    def test_coor_fn(self):
        # test case 1
        pt1 = np.array([0.0, 0.0, 0.0])
        pt2 = np.array([1.0, 0.0, 0.0])
        pt3 = pt1
        pt4 = pt2
        test_val = utils.correlation(pt1, pt2, pt3, pt4)
        np.testing.assert_array_equal(test_val, 1.0)
        # test case 2
        pt2 = np.array([2.0, 0.0, 0.0])
        pt3 = np.array([2.5, 1.0, 0.0])
        pt4 = np.array([4.5, 1.5, 0.0])
        test_val = utils.correlation(pt1, pt2, pt3, pt4)
        np.testing.assert_almost_equal(test_val, 0.970142500145332)

    def test_RB_potential(self):
        value = 3.281250000000001
        test_val = utils.RB_potential(120.0, 5.0, 4.0, 3.0, 5.0, 4.0, 3.0)
        np.testing.assert_almost_equal(value, test_val)

    def test_boltz_dist(self):
        energies = [0.019, 0.15, 0.23, 0.026]
        temp = 300.0
        values = np.array([0.56517132, 0.00356022, 0.00016126, 0.43110720])
        test_vals = utils.boltz_dist(temp, energies)
        np.testing.assert_almost_equal(values, test_vals)

if __name__ == '__main__':
    unittest.main()
