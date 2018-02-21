from pymatgen.io.qchem import QcOutput
import os
import sys
from scipy.optimize import curve_fit
import utils
import numpy as np


class ImportDihedralPotential(object):

    def __init__(self, directory, dihedral_angles, temp=None):
        """
        directory: where all the qchem out files are stored
        angles: list of angles to evaluate energies using RB function

        """
        self.dir = directory
        self.angles = dihedral_angles
        self.abs_energy, self.dihedral, self.molecule, self.mull_charge, self.error = self._get_properties()
        self.energy = utils.relative_energy(self.abs_energy)
        self.params, self.covar = curve_fit(utils.RB_potential, self.dihedral, self.energy)
        self.RB_energy = [utils.RB_potential(angle, *self.params) for angle in self.angles]
        if temp:
            self.temp = temp
            self._get_boltzmann()

    def _get_properties(self):
        energy, dihedral, molecule, mull_charge, errors = [], [], [], [], []
        for f in os.listdir(self.dir):
            if ".qcout" in f:
                if ".orig" not in f:
                    try:
                        output = QcOutput('{d}/{f}'.format(d=self.dir, f=f))
                        finish = output.data[-1]['gracefully_terminated']
                        if not finish:
                            continue
                    except:
                        e = sys.exc_info()[0]
                        errors.append('output/termination {e} in {f}'.format(e=e, f=f))
                        continue
                    try:
                        energy.append(output.final_energy)
                    except:
                        e = sys.exc_info()[0]
                        errors.append('energy {e} in {f}'.format(e=e, f=f))
                        continue
                    try:
                        qchem_in = output.data[-1]['input']
                    except:
                        e = sys.exc_info()[0]
                        errors.append('input {e} in {f}'.format(e=e, f=f))
                    try:
                        mull_charge.append(output.data[-1]['charges']['mulliken'])
                    except:
                        e = sys.exc_info()[0]
                        errors.append('charges {e} in {f}'.format(e=e, f=f))
                    try:
                        molecule.append(output.final_structure)
                    except:
                        e = sys.exc_info()[0]
                        errors.append('final structure {e} in {f}'.format(e=e, f=f))
                    try:
                        constraints = qchem_in.params['opt']['CONSTRAINT']
                        for l in constraints:
                            if 'tors' in l:
                                dihedral.append(l[len(l) - 1])
                    except:
                        e = sys.exc_info()[0]
                        errors.append('constraints {e} in {f}'.format(e=e, f=f))
        return energy, dihedral, molecule, mull_charge, errors

    def _get_boltzmann(self):
        self.prob = utils.boltz_dist(self.temp, self.RB_energy)
        self.cum_prob = [sum(self.prob[0:prob_i]) for prob_i in range(len(self.prob))]
        self.prob_angle = zip(self.cum_prob, self.angles)
