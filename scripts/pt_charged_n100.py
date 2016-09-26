import import_qchem
import utils
import numpy as np
from scipy.optimize import curve_fit
from polymer_chain import Polymer

# neutral chain
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

# charged chain
# import
c_energy, c_dihedral, c_errors = import_qchem.get_energy_dihedral('./charged_dft_b3lyp_out/')
# fit dihedral potential curve
c_rel_eV_energy = utils.relative_energy(c_energy)
c_params, c_covar = curve_fit(utils.RB_potential, c_dihedral, c_rel_eV_energy)
# create list of angles and corresponding energies
c_angles = np.linspace(-180, 180, 3600)
c_RB_energy = [utils.RB_potential(angle, *c_params) for angle in c_angles]
# Boltzmann distribution
c_prob_700 = utils.boltz_dist(700.0, c_RB_energy)
# cumulative probability
c_cum_prob = [sum(c_prob_700[0:prob_i]) for prob_i in range(len(c_prob_700))]
c_prob_angle = np.array(zip(c_cum_prob, c_angles))

# Sampling
monomer_num = 100
monomer_len = 2.548
link_len = 1.480
link_angle = 15.0
sample_num = 15000
electrons = 1
pt = Polymer(monomer_num, monomer_len, link_len, link_angle, n_prob_angle, sample_num, c_prob_angle=c_prob_angle)
c_ave_ete, c_ave_corr = pt.sample_charged_chains(electrons)

# write to file
ete = [c_ave_ete]
utils.write_json(ete, "pt_c_m{m}_t700_{s}_ete.json".format(m=monomer_num, s=sample_num))
corr = c_ave_corr.tolist()
utils.write_json(corr, "pt_c_m{m}_t700_{s}_corr.json".format(m=monomer_num, s=sample_num))
