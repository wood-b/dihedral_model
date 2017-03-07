import numpy as np
from scipy.optimize import curve_fit
from utils import import_qchem
from utils import utils
from core.polymer_chain import Polymer
from core.polymer_chain import RandomChargePolymer
from argparse import ArgumentParser


def run_partial_order_param():

    description = "command line interface for running dihedral_model"
    parser = ArgumentParser(description=description)
    parser.add_argument('-nf', action='store', type=str, help='input file', required=True)
    parser.add_argument('-cf', action='store', type=str, help='input file', required=True)
    parser.add_argument('-gp', action='store', type=int, default=3600,
                        help='number of grid points used for dihedral angles')
    parser.add_argument('-t', action='store', type=float, required=True, help='temperature in kelvin')
    parser.add_argument('-mn', action='store', type=int, required=True, help='monomer_number')
    parser.add_argument('-ml', action='store', type=float, default=2.548, help='monomer_length')
    parser.add_argument('-ll', action='store', type=float, default=1.480, help='link_length')
    parser.add_argument('-la', action='store', type=float, default=15.0, help='link_angle')
    parser.add_argument('-sn', action='store', type=int, required=True, help='sample_number')
    parser.add_argument('-pe', action='store', type=int, required=True, help='percent excited')
    parser.add_argument('-es', action='store', type=int, required=True, help='excitation size')
    parser.add_argument('-o', action='store', type=str, required=True, help='output filename')
    parser.add_argument('-od', action='store', type=str, required=True, help='output directory')

    args = parser.parse_args()

    # import
    energy, dihedral, errors = import_qchem.get_energy_dihedral(args.nf)
    c_energy, c_dihedral, c_errors = import_qchem.get_energy_dihedral(args.cf)
    # fit dihedral potential curve
    rel_eV_energy = utils.relative_energy(energy)
    c_rel_eV_energy = utils.relative_energy(c_energy)
    params, covar = curve_fit(utils.RB_potential, dihedral, rel_eV_energy)
    c_params, c_covar = curve_fit(utils.RB_potential, c_dihedral, c_rel_eV_energy)
    # create list of angles and corresponding energies
    angles = np.linspace(-179.9, 180.0, args.gp)
    RB_energy = [utils.RB_potential(angle, *params) for angle in angles]
    c_RB_energy = [utils.RB_potential(angle, *c_params) for angle in angles]
    # Boltzmann distribution
    prob = utils.boltz_dist(args.t, RB_energy)
    c_prob = utils.boltz_dist(args.t, c_RB_energy)
    # cumulative probability
    cum_prob = [sum(prob[0:prob_i]) for prob_i in range(len(prob))]
    prob_angle = np.array(zip(cum_prob, angles))
    c_cum_prob = [sum(c_prob[0:c_prob_i]) for c_prob_i in range(len(c_prob))]
    c_prob_angle = np.array(zip(c_cum_prob, angles))
    # run dihedral model
    if args.pe == 0:
        poly = Polymer(args.mn, args.ml, args.ll, args.la, prob_angle, args.sn)
        poly.sample_chains()
    if args.pe == 100:
        poly = Polymer(args.mn, args.ml, args.ll, args.la, c_prob_angle, args.sn)
        poly.sample_chains()
    if args.pe != 0 and args.pe != 100:
        poly = RandomChargePolymer(args.mn, args.ml, args.ll, args.la, prob_angle, c_prob_angle, args.sn)
        poly.sample_charged_chains(args.pe, args.es)

    # write files
    for attr, value in poly.__dict__.iteritems():
        if attr.startswith('c_ete_stats'):
            utils.write_json(value.mean, "{dir}/{name}_m{m}_t{t}_{s}_pe{p}_{d}_mean.json".format(
                dir=args.od, name=args.o, m=args.mn, t=args.t, s=args.sn, p=args.pe, d=attr))
            utils.write_json(value.std_error, "{dir}/{name}_m{m}_t{t}_{s}_pe{p}_{d}_std_error.json".format(
                dir=args.od, name=args.o, m=args.mn, t=args.t, s=args.sn, p=args.pe, d=attr))
        if attr.startswith('s_order_param'):
            utils.write_json(value.mean, "{dir}/{name}_m{m}_t{t}_{s}_pe{p}_{d}_mean.json".format(
                dir=args.od, name=args.o, m=args.mn, t=args.t, s=args.sn, p=args.pe, d=attr))
            utils.write_json(value.variance, "{dir}/{name}_m{m}_t{t}_{s}_pe{p}_{d}_var.json".format(
                dir=args.od, name=args.o, m=args.mn, t=args.t, s=args.sn, p=args.pe, d=attr))
            utils.write_json(value.std_error, "{dir}/{name}_m{m}_t{t}_{s}_pe{p}_{d}_std_error.json".format(
                dir=args.od, name=args.o, m=args.mn, t=args.t, s=args.sn, p=args.pe, d=attr))
        if attr.startswith('actual_percent_excited'):
            utils.write_json(value, "{dir}/{name}_m{m}_t{t}_{s}_pe{p}_{d}.json".format(
                dir=args.od, name=args.o, m=args.mn, t=args.t, s=args.sn, p=args.pe, d=attr))

if __name__ == '__main__':
    run_partial_order_param()
