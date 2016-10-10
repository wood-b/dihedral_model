import numpy as np
import math
import json

__author__ = "Brandon Wood"


# Module with an assortment of functions see
# individual function comments for description


# calc and return unit vector of two pts, pts must numpy format
def unit_vector(pt1, pt2):
    line = pt2 - pt1
    return line / math.sqrt((np.dot(line, line)))


# rotate a point counterclockwise around an arbitrary unit vector
# http://inside.mines.edu/fs_home/gmurray/ArbitraryAxisRotation/
# angles range from -180 to 180 not 0 to 360
def point_rotation(pt, angle_degree, unit_vec, origin=None):
    angle = np.pi + (angle_degree * (np.pi / 180.0))
    x, y, z = pt[0], pt[1], pt[2]
    u, v, w = unit_vec[0], unit_vec[1], unit_vec[2]
    if origin is not None:
        a, b, c = origin[0], origin[1], origin[2]
    else:
        a, b, c = 0.0, 0.0, 0.0
    new_x = ((a * (v ** 2 + w ** 2) - u * (b * v + c * w - u * x - v * y - w * z)) * (1 - math.cos(angle))
             + x * math.cos(angle)
             + (-c * v + b * w - w * y + v * z) * math.sin(angle))

    new_y = ((b * (u ** 2 + w ** 2) - v * (a * u + c * w - u * x - v * y - w * z)) * (1 - math.cos(angle))
             + y * math.cos(angle)
             + (c * u - a * w + w * x - u * z) * math.sin(angle))

    new_z = ((c * (u ** 2 + v ** 2) - w * (a * u + b * v - u * x - v * y - w * z)) * (1 - math.cos(angle))
             + z * math.cos(angle)
             + (-b * u + a * v - v * x + u * y) * math.sin(angle))
    return np.array([new_x, new_y, new_z])


def eV_to_kJmol(energy):
    kJ_mol = [(i * 96.48533646) for i in energy]
    return kJ_mol


def eV_to_kcalmol(energy):
    kcal_mol = [(i * 23.06054887) for i in energy]
    return kcal_mol


def relative_energy(energy):
    minimum = min(energy, key=float)
    rel_energy = [i - minimum for i in energy]
    return rel_energy


# correlation function
def correlation(pt1, pt2, pt3, pt4):
    return (np.dot(pt2 - pt1, pt4 - pt3) /
            (math.sqrt(np.dot((pt2 - pt1), (pt2 - pt1)))
             * math.sqrt(np.dot((pt4 - pt3), (pt4 - pt3)))))


# Ryckaert_Bellemans dihedral potential function
def RB_potential(x, a, b, c, d, e, f):
    return (a * 1.0 + b * np.cos(x * np.pi / 180.0)
            + c * (np.cos(x * np.pi / 180.0) ** 2)
            + d * (np.cos(x * np.pi / 180.0) ** 3)
            + e * (np.cos(x * np.pi / 180.0) ** 4)
            + f * (np.cos(x * np.pi / 180.0) ** 5))


# Boltzmann distribution
def boltz_dist(temp_K, energies):
    # kbT in eV/KS
    kb_eV_K = 8.61733238 * 10 ** -5
    kbT = temp_K * kb_eV_K
    # normalization
    boltz_factor = np.array([np.exp(-energy / kbT) for energy in energies])
    normalize_val = sum(boltz_factor)
    prob = boltz_factor / normalize_val
    return prob


def write_json(write_list, filename):
    with open(filename, 'w') as f:
        json.dump(write_list, f)


def read_json(filename):
    with open(filename, 'r') as f:
        read_list = json.load(f)
    assert isinstance(read_list, list)
    return read_list
