import numpy as np
from scipy.optimize import curve_fit
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


# planarity function based on dot product
def planarity(pt1, pt2, pt3, pt4, pt5, pt6):
    vec_1 = (pt2 - pt1)
    vec_2 = (pt2 - pt3)
    normal_1 = np.cross(vec_1, vec_2)
    normal_1 /= np.sqrt(np.dot(normal_1, normal_1))
    vec_3 = pt5 - pt4
    vec_4 = pt5 - pt6
    normal_2 = np.cross(vec_3, vec_4)
    normal_2 /= np.sqrt(np.dot(normal_2, normal_2))
    return np.dot(normal_1, normal_2)


# p2 order parameter often used in liquid crystals
def p2_unit_vec(pt1, pt2, pt3):
    vec_1 = (pt2 - pt1)
    vec_2 = (pt2 - pt3)
    normal = np.cross(vec_1, vec_2)
    normal /= np.sqrt(np.dot(normal, normal))
    return


def p2_ref(pt1, pt2, pt3):
    vec_1 = (pt2 - pt1)
    vec_2 = (pt2 - pt3)
    normal = np.cross(vec_1, vec_2)
    normal /= np.sqrt(np.dot(normal, normal))
    return np.arccos(normal[2])


def p2_order(ref, n_angle):
    theta = n_angle - ref
    return ((3./2.) * (np.cos(theta) ** 2)) - (1./2.)


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


# https://terpconnect.umd.edu/~toh/models/ErrorPropagation.pdf
# Error propagation for multiplication and division
def error_m_d(x, p, sigma_p, q=1, sigma_q=1):
    return x * np.sqrt((sigma_p / p) ** 2 * (sigma_q / q) ** 2)


# Error propagation for natural log
def error_nl(p, sigma_p):
    return sigma_p / p


def poly1d(x, a, b):
    return a * x + b


# Persistence length ln fit
def pt_persist_length(x_vals, corr, std_corr):
    ln_corr = np.log(corr)
    std_ln_corr = error_nl(corr, std_corr)
    popt, pcov = curve_fit(poly1d, x_vals, ln_corr, sigma=std_ln_corr, absolute_sigma=True)
    # dimensionless persistence length
    pt_np = -1.0 / popt[0]
    # persistence length in nanometers
    h = np.sqrt(2.548 ** 2 + 1.480 ** 2 - (2 * 2.548 * 1.480 * np.cos(165.0 * np.pi / 180.0))) / 10.0
    lp = pt_np * h
    # error in persistence length
    perr = np.sqrt(np.diag(pcov))
    # division step
    np_std = error_m_d(pt_np, popt[0], perr[0])
    #np_std = 1.0 / perr[0]
    # multiplication
    lp_std = np_std * h
    return lp, lp_std


def exp_decay(x, a):
    return np.exp(-x / a)


def pt_persist_len_expfit(x_vals, corr, std_error):
    popt, pcov = curve_fit(exp_decay, x_vals, corr,
                           sigma=std_error, absolute_sigma=True)
    pt_np = popt[0]
    h = np.sqrt(2.548 ** 2 + 1.480 ** 2 - (2 * 2.548 * 1.480 * np.cos(165.0 * np.pi / 180.0))) / 10.0
    lp = pt_np * h
    # error
    new_std_error = np.sqrt(np.diag(pcov[0])).flatten() * h
    return lp, new_std_error


def write_json(write_list, filename):
    with open(filename, 'w') as f:
        json.dump(write_list, f)


def read_json(filename):
    with open(filename, 'r') as f:
        read_list = json.load(f)
    return read_list
