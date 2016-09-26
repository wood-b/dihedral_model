import import_qchem
import utils
import numpy as np
from scipy.optimize import curve_fit
from polymer_chain import Polymer

# import
n_energy, n_dihedral, n_errors = import_qchem.get_energy_dihedral('./pt_cust_dft_full')
# fit dihedral potential curve
n_rel_eV_energy = utils.relative_energy(n_energy)
n_params, n_covar = curve_fit(utils.RB_potential, n_dihedral, n_rel_eV_energy)
# create list of angles and corresponding energies
n_angles = np.linspace(-180, 180, 3600)
n_RB_energy = [utils.RB_potential(angle, *n_params) for angle in n_angles]
# Boltzmann distribution
n_prob_700 = utils.boltz_dist(700.0, n_RB_energy)
# cumulative probability
n_cum_prob = [sum(n_prob_700[0:prob_i]) for prob_i in range(len(n_prob_700))]
n_prob_angle = np.array(zip(n_cum_prob, n_angles))

# Sampling
monomer_num = 100
monomer_len = 2.548
link_len = 1.480
link_angle = 15.0
sample_num = 20000
pt = Polymer(monomer_num, monomer_len, link_len, link_angle, n_prob_angle, sample_num)
n_ave_ete, n_ave_corr = pt.sample_neutral_chains()

# write to file
ete = n_ave_ete.tolist()
utils.write_json(ete, "pt_n_m{m}_t700_{s}_ete.json".format(m=monomer_num, s=sample_num))
corr = n_ave_corr.tolist()
utils.write_json(corr, "pt_n_m{m}_t700_{s}_corr.json".format(m=monomer_num, s=sample_num))
