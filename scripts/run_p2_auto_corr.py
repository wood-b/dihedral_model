import numpy as np
from utils import utils
from core.polymer_chain import Polymer
from argparse import ArgumentParser


def run_p2_auto_corr():

    description = "command line interface for running p2 autocorrelation function"
    parser = ArgumentParser(description=description)
    parser.add_argument('-gf', action='store', type=str, default=None, help='ground state prob angle JSON file')
    parser.add_argument('-cf', action='store', type=str, default=None, help='charged or excited prob angle JSON file')
    parser.add_argument('-mn', action='store', type=int, required=True, help='monomer_number')
    parser.add_argument('-ml', action='store', type=float, default=2.548, help='monomer_length')
    parser.add_argument('-ll', action='store', type=float, default=1.480, help='link_length')
    parser.add_argument('-la', action='store', type=float, default=15.0, help='link_angle')
    parser.add_argument('-sn', action='store', type=int, required=True, help='sample_number')
    parser.add_argument('-o', action='store', type=str, required=True, help='output filename')
    parser.add_argument('-od', action='store', type=str, required=True, help='output directory')

    args = parser.parse_args()

    # import ground state - Probability angle
    if args.gf:
        prob_angle = np.array(utils.read_json(args.gf))
    # import excited - Probability angle
    if args.cf:
        prob_angle = np.array(utils.read_json(args.cf))

    # run dihedral model
    poly = Polymer(args.mn, args.ml, args.ll, args.la, prob_angle, args.sn)
    for chain_i in range(1, args.sn + 1, 1):
        poly.rotate_chain()
        poly.p2_auto_corr(poly.relax_chain)

    # write files
    run_dict = {}
    for attr, value in poly.__dict__.iteritems():
        if attr.startswith('s_x_corr'):
            s_x_corr_dict = {'mean': [i.mean for i in value],
                             'variance': [i.variance for i in value],
                             'std_error': [i.std_error for i in value]}
            run_dict['s_x_corr'] = s_x_corr_dict

    utils.write_json(run_dict, "{dir}/{name}_mn{m}_sn{s}_p2_corr.json".format(
                dir=args.od, name=args.o, m=args.mn, s=args.sn))

if __name__ == '__main__':
    run_p2_auto_corr()
