import numpy as np
from scipy.optimize import curve_fit
from utils import import_qchem
from utils import utils
from core.polymer_chain import Polymer
from argparse import ArgumentParser


def run_p2_auto_corr():

    description = "command line interface for running p2 autocorrelation function"
    parser = ArgumentParser(description=description)
    parser.add_argument('-f', action='store', type=str, help='input file', required=True)
    parser.add_argument('-gp', action='store', type=int, default=3600,
                        help='number of grid points used for dihedral angles')
    parser.add_argument('-t', action='store', type=float, required=True, help='temperature in kelvin')
    parser.add_argument('-mn', action='store', type=int, required=True, help='monomer_number')
    parser.add_argument('-ml', action='store', type=float, default=2.548, help='monomer_length')
    parser.add_argument('-ll', action='store', type=float, default=1.480, help='link_length')
    parser.add_argument('-la', action='store', type=float, default=15.0, help='link_angle')
    parser.add_argument('-sn', action='store', type=int, required=True, help='sample_number')
    parser.add_argument('-o', action='store', type=str, required=True, help='output filename')
    parser.add_argument('-od', action='store', type=str, required=True, help='output directory')

    args = parser.parse_args()

    # import
    energy, dihedral, errors = import_qchem.get_energy_dihedral(args.f)
    # fit dihedral potential curve
    rel_eV_energy = utils.relative_energy(energy)
    params, covar = curve_fit(utils.RB_potential, dihedral, rel_eV_energy)
    # create list of angles and corresponding energies
    angles = np.linspace(-179.9, 180.0, args.gp)
    RB_energy = [utils.RB_potential(angle, *params) for angle in angles]
    # Boltzmann distribution
    prob = utils.boltz_dist(args.t, RB_energy)
    # cumulative probability
    cum_prob = [sum(prob[0:prob_i]) for prob_i in range(len(prob))]
    prob_angle = np.array(zip(cum_prob, angles))
    # run dihedral model
    poly = Polymer(args.mn, args.ml, args.ll, args.la, prob_angle, args.sn)
    for chain_i in range(1, args.sn + 1, 1):
        poly.rotate_chain()
        poly.p2_auto_corr()

    # write files
    for attr, value in poly.__dict__.iteritems():
        if attr.startswith('s_x_corr'):
            mean_list = [i.mean for i in value]
            utils.write_json(mean_list, "{dir}/{name}_m{m}_t{t}_{s}_{d}_mean.json".format(
                dir=args.od, name=args.o, m=args.mn, t=args.t, s=args.sn, d=attr))
            var_list = [i.variance for i in value]
            utils.write_json(var_list, "{dir}/{name}_m{m}_t{t}_{s}_{d}_var.json".format(
                dir=args.od, name=args.o, m=args.mn, t=args.t, s=args.sn, d=attr))

if __name__ == '__main__':
    run_p2_auto_corr()
