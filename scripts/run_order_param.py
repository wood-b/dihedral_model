import numpy as np
from utils import utils
from core.polymer_chain import Polymer
from core.polymer_chain import RandomChargePolymer
from argparse import ArgumentParser


def run_order_param():

    description = "command line interface for running dihedral_model"
    parser = ArgumentParser(description=description)
    parser.add_argument('-gf', action='store', type=str, default=None, help='ground state prob angle JSON file')
    parser.add_argument('-cf', action='store', type=str, default=None, help='charged or excited prob angle JSON file')
    parser.add_argument('-mn', action='store', type=int, required=True, help='monomer_number')
    parser.add_argument('-ml', action='store', type=float, default=2.47, help='monomer_length')
    parser.add_argument('-ll', action='store', type=float, default=1.46, help='link_length')
    parser.add_argument('-la', action='store', type=float, default=15.2, help='link_angle')
    parser.add_argument('-et', action='store', type=str, default=None, help='excitation type')
    parser.add_argument('-sn', action='store', type=int, required=True, help='sample_number')
    parser.add_argument('-nc', action='store', type=int, required=True, help='number of charged or excited dihedrals')
    parser.add_argument('-o', action='store', type=str, required=True, help='output filename')
    parser.add_argument('-od', action='store', type=str, required=True, help='output directory')

    args = parser.parse_args()

    # What type of excitation
    c_ml = None
    c_ll = None
    c_la = None
    if args.et == 'trip':
        c_ml = 2.50
        c_ll = 1.35
        c_la = 14.1
    if args.et == "cat":
        c_ml = 2.45
        c_ll = 1.40
        c_la = 14.3

    # import ground state - Probability angle
    if args.gf:
        prob_angle = np.array(utils.read_json(args.gf))
    # import excited - Probability angle
    if args.cf:
        c_prob_angle = np.array(utils.read_json(args.cf))

    # run dihedral model
    if args.nc == 0:
        poly = Polymer(args.mn, args.ml, args.ll, args.la, prob_angle, args.sn)
        for chain_i in range(1, args.sn + 1, 1):
            poly.rotate_chain()
            poly.p2_order_param(poly.relax_chain)

    if args.nc == (args.mn - 1):
        poly = Polymer(args.mn, c_ml, c_ll, c_la, c_prob_angle, args.sn)
        for chain_i in range(1, args.sn + 1, 1):
            poly.rotate_chain()
            poly.p2_order_param(poly.relax_chain)

    if args.nc != 0 and args.nc != (args.mn - 1):
        poly = RandomChargePolymer(args.mn, args.ml, args.ll, args.la, prob_angle,
                                   c_ml, c_ll, c_la, c_prob_angle, args.sn)
        for chain_i in range(1, args.sn + 1, 1):
            poly.shuffle_charged_chain(args.nc)
            poly.p2_order_param(poly.charged_chain)

    # write files
    run_dict = {}
    run_dict['monomer_number'] = args.mn
    run_dict['charged_dihedral_number'] = args.nc
    run_dict['sample_number'] = args.sn
    run_dict['excitation_type'] = args.et
    for attr, value in poly.__dict__.iteritems():
        if attr.startswith('s_order_param'):
            s_order_param_dict = {'mean': value.mean, 'variance': value.variance, 'std_error': value.std_error}
            run_dict['s_order_param'] = s_order_param_dict

    utils.write_json(run_dict, "{dir}/{name}_mn{m}_nc{nc}_sn{s}_s_order_param.json".format(dir=args.od,
                                                                                           name=args.o,
                                                                                           m=args.mn,
                                                                                           nc=args.nc,
                                                                                           s=args.sn))


if __name__ == '__main__':
    run_order_param()
