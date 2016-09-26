from pymatgen.io.babel import BabelMolAdaptor
import numpy as np
import pybel as pb
import math

# get dihedral angle
def get_dihedral(mol, site_indexes):
    babe_mol = pb.Molecule(BabelMolAdaptor(mol).openbabel_mol)
    dihedral_val = babe_mol.OBMol.GetTorsion(site_indexes[0], site_indexes[1], site_indexes[2], site_indexes[3])
    return dihedral_val

# set dihedral angle   
def set_dihedral(mol, site_indexes, angle, write_xyz=False):
    babe_mol = pb.Molecule(BabelMolAdaptor(mol).openbabel_mol)
    babe_mol.OBMol.SetTorsion(site_indexes[0], site_indexes[1], site_indexes[2], site_indexes[3], angle*math.pi/180.)
    if write_xyz:
        babe_mol.write("xyz", filename="rotated{angle}.xyz".format(angle=angle), overwrite=False)
    rotated_mol = BabelMolAdaptor(babe_mol.OBMol).pymatgen_mol
    return rotated_mol
    
# dihedral scan
def dihedral_scan(mol, site_indexes, start, stop, num, write_xyz=False):
    angles = np.linspace(start, stop, num)
    mol_dict = {angle: set_dihedral(mol, site_indexes, angle, write_xyz) for angle in angles}
    return mol_dict