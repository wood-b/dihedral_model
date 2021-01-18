from pymatgen.io.qchem.outputs import QCOutput
from pymatgen.core.units import Energy
import os
import sys
from scipy.optimize import curve_fit
from utils import utils


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
            if any(item in f for item in [".qcout", ".out"]):
                output = QCOutput('{d}/{f}'.format(d=self.dir, f=f))
                # if calc finished get properties
                if output.data.get("completion", []):
                    # energy in eV
                    energy.append(Energy(output.data["final_energy"], "Ha").to("eV"))
                    # dihedral
                    if output.data.get("opt_constraint"):
                        if "Dihedral" in output.data["opt_constraint"]:
                            dihedral.append(float(output.data["opt_constraint"][-1]))
                        else:
                            dihedral.append("No dihedral constraint or multiple constraints check output")
                    # molecule from final structure
                    molecule.append(output.data["molecule_from_optimized_geometry"])
                    # mulliken charges
                    mull_charge.append(output.data["Mulliken"])
                # errors
                errors.append(output.data["errors"])
        return energy, dihedral, molecule, mull_charge, errors

    def _get_boltzmann(self):
        self.prob = utils.boltz_dist(self.temp, self.RB_energy)
        self.cum_prob = [sum(self.prob[0:prob_i]) for prob_i in range(len(self.prob))]
        self.prob_angle = [list(i) for i in zip(self.cum_prob, self.angles)]
