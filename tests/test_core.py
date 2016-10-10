import numpy as np
from core.polymer_chain import Polymer
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
        np.testing.assert_almost_equal(self.polymer.ete_stats.means[self.monomer_num], mean_ete, decimal=2)
        np.testing.assert_almost_equal(self.polymer.ete_stats.stdevs[self.monomer_num], std_ete, decimal=2)
        np.testing.assert_almost_equal(self.polymer.corr_stats.means[self.monomer_num - 1], mean_corr, decimal=2)
        np.testing.assert_almost_equal(self.polymer.corr_stats.stdevs[self.monomer_num - 1], std_corr, decimal=2)


if __name__ == '__main__':
    unittest.main()
