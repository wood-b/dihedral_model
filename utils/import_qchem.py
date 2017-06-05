from pymatgen.io.qchem import QcOutput
import os
from scipy.optimize import curve_fit
import utils


class ImportDihedralPotential(object):

    def __init__(self, directory, dihedral_angles):
        """
        directory: where all the qchem out files are stored
        angles: list of angles to evaluate energies using RB function

        """
        self.dir = directory
        self.angles = dihedral_angles
        self.abs_energy, self.dihedral, self.error = self._get_energy_dihedral()
        self.energy = utils.relative_energy(self.abs_energy)
        self.params, self.covar = curve_fit(utils.RB_potential, self.dihedral, self.energy)
        self.RB_energy = [utils.RB_potential(angle, *self.params) for angle in self.angles]

    def _get_energy_dihedral(self):
        energy, dihedral, errors = [], [], []
        for f in os.listdir(self.dir):
            if ".qcout" in f:
                if ".orig" not in f:
                    try:
                        output = QcOutput('{d}/{f}'.format(d=self.dir, f=f))
                        qchem_in = output.data[-1]['input']
                        try:
                            energy.append(output.final_energy)
                            constraints = qchem_in.params['opt']
                            for l in constraints:
                                if 'tors' in l:
                                    dihedral.append(l[len(l) - 1])
                        except IndexError:
                            errors.append('no energy in {f}'.format(f=f))
                    except AttributeError:
                        errors.append('unknown error in {f}'.format(f=f))
        return energy, dihedral, errors
