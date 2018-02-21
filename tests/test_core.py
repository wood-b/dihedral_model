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
        cls.monomer_num = 25
        cls.monomer_len = 1.5
        cls.link_len = 0.5
        cls.link_angle = 15.0
        cls.sample_num = 1000
        cls.prob_angle = np.array([[0.0, -180.0], [0.2, -90.0], [0.3, -45.0], [0.4, 0.0],
                                   [0.5, 45.0], [0.6, 90.0], [0.8, 180.0]])
        # create polymer
        cls.polymer = Polymer(cls.monomer_num, cls.monomer_len, cls.link_len, cls.link_angle,
                              cls.prob_angle, cls.sample_num)
        # manually calculated bead or atom positions
        # the first commented out positions are for when l1 is first instead of l2
        """cls.linear_pos_values = np.array([[0.0, 0.0, 0.0], [1.5, 0.0, 0.0],
                                         [1.9829629131445341, -0.12940952255126037, 0.0],
                                         [3.4829629131445339, -0.12940952255126037, 0.0],
                                         [3.9659258262890682, 0.0, 0.0],
                                         [5.4659258262890678, 0.0, 0.0],
                                         [5.9488887394336016, -0.12940952255126037, 0.0],
                                         [7.4488887394336016, -0.12940952255126037, 0.0]])"""
        cls.linear_chain_actual = np.array([[0.0, 0.0, 0.0], [1.5, 0.0, 0.0],
                                            [1.9829629131445341, 0.12940952255126037, 0.0],
                                            [3.4829629131445339, 0.12940952255126037, 0.0],
                                            [3.9659258262890682, 0.0, 0.0],
                                            [5.4659258262890678, 0.0, 0.0],
                                            [5.9488887394336016, 0.12940952255126037, 0.0],
                                            [7.4488887394336016, 0.12940952255126037, 0.0]])

    def test_build_chain(self):
        np.testing.assert_almost_equal(self.linear_chain_actual, self.polymer.chain[:8])

    def test_random_angle(self):
        angle_num = self.monomer_num - 1
        self.polymer.rotate_chain()
        np.testing.assert_equal(angle_num, len(self.polymer.dihedral_set))
        for angle in self.polymer.dihedral_set:
            self.assertIn(angle, self.prob_angle[:, 1])

    def test_rotate_chain(self):
        self.polymer.rotate_chain()
        # this makes a fake molecule and checks all the dihedral angles
        fake_atoms = []
        fake_atom_coords = []
        for coord in self.polymer.relax_chain:
            fake_atoms.append('C')
            fake_atom_coords.append(coord)
        fake_mol = Molecule(species=fake_atoms, coords=fake_atom_coords)
        # find all the dihedral angles
        dihedral_list_actual = []
        for site, val in enumerate(fake_mol, 1):
            if site <= len(fake_mol) - 3 and site % 2 != 0:
                da = round(dihedral_tools.get_dihedral(fake_mol, [site, site + 1, site + 2, site + 3]))
                # this if statement ensures 180 == -180 and 0 == -0
                if da == -180.0 or da == -0.0:
                    da = abs(da)
                dihedral_list_actual.append(da)
        self.assertEqual(len(dihedral_list_actual), len(self.polymer.dihedral_set))
        rotate_chain_dihedral_set = []
        # again this loop ensures 180 == -180 and 0 == -0
        for angle in self.polymer.dihedral_set:
            if angle == -180.0 or angle == -0.0:
                rotate_chain_dihedral_set.append(abs(angle))
            else:
                rotate_chain_dihedral_set.append(angle)
        np.testing.assert_almost_equal(dihedral_list_actual, rotate_chain_dihedral_set)

    def test_tangent_auto_corr(self):
        # check case where all tangent vectors are aligned
        self.polymer.tangent_auto_corr(self.polymer.chain)
        for stat in self.polymer.tangent_corr:
            np.testing.assert_allclose(stat.mean, 1.0)

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
        samples = 200
        p2_polymer = Polymer(self.monomer_num, self.monomer_len, self.link_len, self.link_angle, self.prob_angle)
        p2_polymer.p2_auto_corr(p2_polymer.chain)
        # check correlation is 1 when all aligned
        for stat in p2_polymer.s_x_corr:
            np.testing.assert_allclose(1.0, stat.mean)
        # check the correlation over a bunch of samples
        pair_interacts = int((self.monomer_num * (self.monomer_num + 1)) / 2)
        # adds 1 to all lists for case where everything is aligned
        ensemble_list = [[1.0] for i in range(self.monomer_num)]
        # loops of the number of samples
        for sample in range(1, samples):
            p2_polymer.rotate_chain()
            p2_polymer.p2_auto_corr(p2_polymer.relax_chain)
            polymer_list = []
            for i in range(self.monomer_num):
                pair_list = []
                for j in range(i, self.monomer_num, 1):
                    pair_list.append(((3. / 2.) * (np.dot(p2_polymer.unit_normal[i],
                                                          p2_polymer.unit_normal[j]) ** 2)) - (1. / 2.))
                polymer_list.append(pair_list)
            for l in polymer_list:
                for i, val in enumerate(l):
                    ensemble_list[i].append(val)
        actual_means = [np.mean(i) for i in ensemble_list]

        # check the right number of pair interactions were sampled
        # checks all the self interactions
        np.testing.assert_equal(int((samples * self.monomer_num)), int(p2_polymer.s_x_corr[0].k))
        # checks the longest interaction only 1 per polymer chain sample
        np.testing.assert_equal(int(samples), int(p2_polymer.s_x_corr[-1].k))
        for i, stat in enumerate(p2_polymer.s_x_corr):
            # print(actual_means[i], stat.mean)
            np.testing.assert_allclose(actual_means[i], stat.mean, atol=0.01, rtol=0.0)

    def test_sample_chain(self):
        # sample by looping over rotate_chains
        # start a new chain
        sample_polymer = Polymer(self.monomer_num, self.monomer_len, self.link_len, self.link_angle,
                                 self.prob_angle, sample_num=self.sample_num)
        end_to_end = []
        for i in range(self.sample_num):
            sample_polymer.rotate_chain()
            end_to_end.append(sample_polymer.end_to_end[-1])
        mean_ete = np.mean(end_to_end)
        std_ete = np.std(end_to_end)
        # sample using polymer class
        sample_polymer.sample_chains()
        # print(mean_ete, sample_polymer.ete_stats.mean[-1])
        # print(std_ete, sample_polymer.ete_stats.stdev[-1])
        np.testing.assert_allclose(mean_ete, sample_polymer.ete_stats.mean[-1], atol=0.5, rtol=0.0)
        np.testing.assert_allclose(std_ete, sample_polymer.ete_stats.stdev[-1], atol=0.5, rtol=0.0)


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
        cls.c_prob_angle = np.array([[0.0, 175.0], [0.5, 5.0]])
        cls.c_polymer = RandomChargePolymer(cls.monomer_num, cls.monomer_len, cls.link_len, cls.link_angle,
                                            cls.prob_angle, cls.c_prob_angle, cls.sample_num)

    def test_c_random_angle(self):
        self.c_polymer.shuffle_charged_chain(10)
        c_angle_num = 10
        self.assertEqual(c_angle_num, len(self.c_polymer.c_dihedral_set))
        for angle in self.c_polymer.c_dihedral_set:
            self.assertIn(angle, self.c_prob_angle[:, 1])

    def test_shuffle_charged_chain(self):
        self.c_polymer.shuffle_charged_chain(10)
        # check position lists are same length
        self.assertEqual(len(self.c_polymer.relax_chain), len(self.c_polymer.charged_chain))
        # loop through the chain and check dihedral angles
        fake_atoms = []
        fake_atom_coords = []
        for coord in self.c_polymer.charged_chain:
            fake_atoms.append('C')
            fake_atom_coords.append(coord)
        fake_mol = Molecule(species=fake_atoms, coords=fake_atom_coords)
        # find all the dihedral angles
        dihedral_list_actual = []
        for site, val in enumerate(fake_mol, 1):
            if site <= len(fake_mol) - 3 and site % 2 != 0:
                da = round(dihedral_tools.get_dihedral(fake_mol, [site, site + 1, site + 2, site + 3]))
                # this if statement ensures 180 == -180 and 0 == -0
                if da == -180.0 or da == -0.0:
                    da = abs(da)
                dihedral_list_actual.append(da)
        self.assertEqual(len(dihedral_list_actual), len(self.c_polymer.shuffle_dihedral_set))
        shuffle_dihedral_set = []
        # again this loop ensures 180 == -180 and 0 == -0
        for angle in self.c_polymer.shuffle_dihedral_set:
            if angle == -180.0 or angle == -0.0:
                shuffle_dihedral_set.append(abs(angle))
            else:
                shuffle_dihedral_set.append(angle)
        np.testing.assert_almost_equal(dihedral_list_actual, shuffle_dihedral_set)


if __name__ == '__main__':
    unittest.main()
