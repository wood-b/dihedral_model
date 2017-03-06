import numpy as np
from core.polymer_chain import Polymer
from core.polymer_chain import RandomChargePolymer
from pymatgen import Molecule
from utils import dihedral_tools
import unittest

__author__ = "Brandon Wood"


class TestPolymer(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # setup for polymer class
        cls.monomer_num = 4
        cls.monomer_len = 1.5
        cls.link_len = 0.5
        cls.link_angle = 15.0
        cls.sample_num = 5000
        cls.prob_angle = np.array([[0.0, -180.0], [0.2, -90.0], [0.3, -45.0], [0.4, 0.0],
                                   [0.5, 45.0], [0.6, 90.0], [0.8, 180.0]])
        # create polymer
        cls.polymer = Polymer(cls.monomer_num, cls.monomer_len, cls.link_len, cls.link_angle, cls.prob_angle, cls.sample_num)
        # manually calculated bead or atom positions
        cls.linear_pos_values = np.array([[0.0, 0.0, 0.0], [1.5, 0.0, 0.0],
                                         [1.9829629131445341, -0.12940952255126037, 0.0],
                                         [3.4829629131445339, -0.12940952255126037, 0.0],
                                         [3.9659258262890682, 0.0, 0.0],
                                         [5.4659258262890678, 0.0, 0.0],
                                         [5.9488887394336016, -0.12940952255126037, 0.0],
                                         [7.4488887394336016, -0.12940952255126037, 0.0]])

    def test_build_chain(self):
        np.testing.assert_almost_equal(self.linear_pos_values, self.polymer.chain)

    def test_random_angle(self):
        angle_num = 3
        self.polymer.rotate_chain()
        np.testing.assert_equal(angle_num, len(self.polymer.dihedral_set))
        for angle in self.polymer.dihedral_set:
            self.assertIn(angle, self.prob_angle[:, 1])

    def test_rotate_chain(self):
        self.polymer.rotate_chain()
        # make a fake pymatgen molecule allowing the use of dihedral tools
        fake_atoms = ["C", "C", "C", "C", "C", "C", "C", "C"]
        fake_molecule = Molecule(fake_atoms, self.polymer.relax_chain)
        # check rotated dihedrals
        # angles of 180 and -180 are the same
        circular_vals = [-180.0, 180.0]
        dh_1 = dihedral_tools.get_dihedral(fake_molecule, [1, 2, 3, 4])
        dh_2 = dihedral_tools.get_dihedral(fake_molecule, [3, 4, 5, 6])
        dh_3 = dihedral_tools.get_dihedral(fake_molecule, [5, 6, 7, 8])
        dhs = [dh_1, dh_2, dh_3]
        # run multiple trials to get a mix of random dihedral angles
        for multiple_trials in range(10):
            for i, dh in enumerate(dhs):
                if self.polymer.dihedral_set[i] in circular_vals:
                    self.assertIn(round(dh), circular_vals)
                else:
                    np.testing.assert_almost_equal(self.polymer.dihedral_set[i], dh)

    def test_unit_normal_vectors(self):
        self.polymer._unit_normal_vectors(self.polymer.chain)
        np.testing.assert_array_equal(len(self.polymer.unit_normal), self.monomer_num)
        totally_planar_normal = np.array([0.0, 0.0, 1.0])
        for u_vec in self.polymer.unit_normal:
            np.testing.assert_almost_equal(u_vec ** 2, totally_planar_normal ** 2)
        self.polymer._unit_normal_vectors(self.polymer.relax_chain)
        for u_vec in self.polymer.unit_normal:
            np.testing.assert_almost_equal(np.linalg.norm(u_vec), 1.0)
        calc_u_vectors = np.zeros((self.monomer_num, 3))
        index = 0
        for i, pt in enumerate(self.polymer.relax_chain):
            if i == 0:
                vec1 = self.polymer.relax_chain[i + 1] - pt
                vec2 = self.polymer.relax_chain[i + 2] - pt
                calc_u_vectors[i] = np.cross(vec1, vec2)
                calc_u_vectors[i] /= np.linalg.norm(calc_u_vectors[i])
                index += 1
            if i % 2 != 0 and i < (len(self.polymer.relax_chain) - 2):
                vec1 = self.polymer.relax_chain[i + 1] - pt
                vec2 = self.polymer.relax_chain[i + 2] - pt
                calc_u_vectors[index] = np.cross(vec1, vec2)
                calc_u_vectors[index] /= np.linalg.norm(calc_u_vectors[index])
                index += 1
        np.testing.assert_almost_equal(self.polymer.unit_normal ** 2, calc_u_vectors ** 2)

    def test_p2_order_param(self):
        # two case all aligned, and isotropic
        # case 1 all aligned
        z_unit = np.array([0., 0., 1.] * 1000)
        z_unit.shape = (1000, 3)
        self.polymer.p2_order_param(unit_vectors=z_unit)
        np.testing.assert_almost_equal(np.trace(self.polymer.director_matrix), 0.0)
        np.testing.assert_almost_equal(self.polymer.s_order_param.mean, 1.0)
        # case 2 isotropic
        # generate uniform vectors on a unit sphere
        index = 0
        n = 50000
        iso_unit = np.zeros((n, 3))
        while index <= (n - 1):
            chi_1 = np.random.uniform(0.0, 1.0, 1)
            chi_2 = np.random.uniform(0.0, 1.0, 1)
            xhi_1 = 1 - (2 * chi_1)
            xhi_2 = 1 - (2 * chi_2)
            xhi_sq = xhi_1 ** 2 + xhi_2 ** 2
            if xhi_sq < 1:
                iso_unit[index] = [2 * xhi_1 * ((1 - xhi_sq) ** (1. / 2.)),
                                    2 * xhi_2 * ((1 - xhi_sq) ** (1. / 2.)),
                                    1 - 2 * xhi_sq]
                index += 1
        self.polymer.p2_order_param(unit_vectors=iso_unit)
        np.testing.assert_almost_equal(np.trace(self.polymer.director_matrix), 0.0)
        np.testing.assert_almost_equal(self.polymer.s_order_param.mean, 0.0, decimal=1)

    def test_p2_auto_corr(self):
        self.polymer.p2_auto_corr(self.polymer.chain)
        # check correlation is 1 when all aligned
        for stat in self.polymer.s_x_corr:
            np.testing.assert_allclose(stat.mean, 1.0)
        # check the correlation
        self.polymer.rotate_chain()
        self.polymer._unit_normal_vectors(self.polymer.relax_chain)
        # since s_x_corr is updated each time p2_auto_corr is called
        # s_x_corr has one entry of 1s from all aligned case
        test_vals = np.zeros((5, 10))
        for i in range(10):
            test_vals[0][i] = 1.
        for loop in range(1, 5, 1):
            self.polymer.rotate_chain()
            self.polymer._unit_normal_vectors(self.polymer.relax_chain)
            self.polymer.p2_auto_corr(self.polymer.relax_chain)
            count = 0
            for i in range(len(self.polymer.unit_normal)):
                for j in range(i, len(self.polymer.unit_normal), 1):
                    test_vals[loop][count] = ((3./2.) * (np.dot(self.polymer.unit_normal[j],
                                                     self.polymer.unit_normal[j - i]) ** 2)) - (1./2.)
                    count += 1
        test_mean = np.array([np.mean(test_vals[:, [0, 1, 2, 3]]), np.mean(test_vals[:, [4, 5, 6]]),
                              np.mean(test_vals[:, [7, 8]]), np.mean(test_vals[:, 9])])

        for i, stat in enumerate(self.polymer.s_x_corr):
            np.testing.assert_allclose(stat.mean, test_mean[i])

    def test_sample_chain(self):
        # sample by looping over rotate_chains
        end_to_end = []
        corr = []
        for i in range(self.sample_num):
            self.polymer.rotate_chain()
            end_to_end.append(self.polymer.end_to_end[self.monomer_num])
            corr.append(self.polymer.corr[self.monomer_num - 1])
        mean_ete = np.mean(end_to_end)
        std_ete = np.std(end_to_end)
        mean_corr = np.mean(corr)
        std_corr = np.std(corr)
        # sample using polymer class
        self.polymer.sample_chains()
        np.testing.assert_almost_equal(self.polymer.ete_stats.mean[self.monomer_num], mean_ete, decimal=1)
        np.testing.assert_almost_equal(self.polymer.ete_stats.stdev[self.monomer_num], std_ete, decimal=1)
        np.testing.assert_almost_equal(self.polymer.corr_stats.mean[self.monomer_num - 1], mean_corr, decimal=1)
        np.testing.assert_almost_equal(self.polymer.corr_stats.stdev[self.monomer_num - 1], std_corr, decimal=1)


class TestRandomChargedPolymer(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.monomer_num = 51
        cls.monomer_len = 1.5
        cls.link_len = 0.5
        cls.link_angle = 15.0
        cls.sample_num = 500
        cls.prob_angle = np.array([[0.0, -180.0], [0.2, -90.0], [0.3, -45.0], [0.4, 0.0],
                                   [0.5, 45.0], [0.6, 90.0], [0.8, 180.0]])
        cls.c_prob_angle = np.array([[0.0, -182.0], [0.5, 2.0]])
        cls.c_polymer = RandomChargePolymer(cls.monomer_num, cls.monomer_len, cls.link_len, cls.link_angle,
                              cls.prob_angle, cls.c_prob_angle, cls.sample_num)

    def test_c_random_angle(self):
        self.c_polymer.rotate_charged_chain(20, 5)
        c_angle_num = 10
        self.assertEqual(c_angle_num, len(self.c_polymer.c_dihedral_set))
        self.c_polymer._pick_links(2, 5)
        self.assertEqual(len(self.c_polymer.c_links), len(self.c_polymer.c_dihedral_set))
        for angle in self.c_polymer.c_dihedral_set:
            self.assertIn(angle, self.c_prob_angle[:, 1])

    def test_pick_links(self):
        # test different number of sites
        self.c_polymer._pick_links(2, 5)
        link_num_2 = len(self.c_polymer.c_links)
        link_num_2_val = 10
        self.assertEqual(link_num_2_val, link_num_2)
        self.c_polymer._pick_links(5, 5)
        link_num_5 = len(self.c_polymer.c_links)
        link_num_5_val = 25
        self.assertEqual(link_num_5_val, link_num_5)
        self.c_polymer._pick_links(9, 5)
        link_num_9 = len(self.c_polymer.c_links)
        link_num_9_val = 45
        self.assertEqual(link_num_9_val, link_num_9)
        for i in range(100):
            self.c_polymer._pick_links(9, 5)
            test = len(np.unique(self.c_polymer.c_links))
            val = 45
            self.assertEqual(val, test)

    def test_rotate_charged_chain(self):
        self.c_polymer.rotate_charged_chain(20, 5)
        # check percent excited
        self.assertEqual(20, self.c_polymer.actual_percent_excited)
        # check position lists are same length
        self.assertEqual(len(self.c_polymer.relax_chain), len(self.c_polymer.charged_chain))
        diff_pos = []
        for i in range(len(self.c_polymer.relax_chain)):
            if any(self.c_polymer.relax_chain[i] != self.c_polymer.charged_chain[i]):
                diff_pos.append(i)
        pos_val = sorted(self.c_polymer.c_links)[0]
        pos_val = (pos_val * 2) + 1
        self.assertEqual(pos_val, diff_pos[0])

if __name__ == '__main__':
    unittest.main()
