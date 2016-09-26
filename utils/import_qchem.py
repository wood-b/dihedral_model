import numpy as np
from pymatgen import Molecule
from pymatgen.io.qchem import QcTask, QcOutput
import os
import re

def get_energy_dihedral(directory):
    energy, dihedral, errors = [], [], []
    for f in os.listdir(directory):
        if ".qcout" in f:
            if ".orig" not in f:
                try:
                    output = QcOutput('{d}/{f}'.format(d=directory,f=f))
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
