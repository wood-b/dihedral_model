import numpy as np
import math
from utils import utils
from utils.stats import Stats, ArrayStats
from utils.histogram import Histogram
from numpy import linalg as la

__author__ = "Brandon Wood"


class Polymer(object):
    def __init__(self, monomer_num, monomer_length, link_length,
                 link_angle, prob_angle, sample_num=None):
        """
        :param monomer_num: number of monomers in polymer chain
        :param monomer_length: length of monomer tangent line
        :param link_length: length of link between monomers
        :param link_angle: angle between monomer tangent and link
        :param prob_angle: array of dihedral probability and angle pairs
        :param sample_num: number of chains to sample
        """
        self.chain = np.zeros(((2 * monomer_num), 3))
        self.relax_chain = np.zeros(((2 * monomer_num), 3))
        self.monomer_num = monomer_num
        self.monomer_len = monomer_length
        self.link_len = link_length
        self.link_angle = (link_angle * np.pi / 180)
        self.prob_angle = prob_angle
        self.sample_num = sample_num
        self.end_to_end = None
        self.corr = None
        self.unit_normal = None
        self.theta = None
        self.director = None
        self.director_matrix = None
        self.s_order_param = Stats()
        self.s_x_corr = [Stats() for i in range(self.monomer_num)]
        self.ete_stats = ArrayStats(self.monomer_num + 1)
        self.tangent_corr = [Stats() for i in range(self.monomer_num)]
        self.dihedral_hist = Histogram(-180.0, 180.0, 361)
        self.ete_hist = []
        # position monomer tangent and links
        self.m = [self.monomer_len, 0, 0]
        self.l1 = [self.link_len * math.cos(self.link_angle),
                   -self.link_len * math.sin(self.link_angle), 0]
        self.l2 = [self.link_len * math.cos(self.link_angle),
                   self.link_len * math.sin(self.link_angle), 0]
        # build all trans chain
        self._build_chain()
        # set of random dihedral angles
        self.dihedral_set = []
        # contour length in Angstroms
        self.contour_length = None

    # builds an all trans chain
    def _build_chain(self):
        link_iter = 'l2'
        for pos in range(1, (2 * self.monomer_num), 1):
            if pos % 2 != 0:
                self.chain[pos] = self.chain[pos - 1] + self.m
                continue
            elif pos % 2 == 0:
                if link_iter == 'l1':
                    self.chain[pos] = self.chain[pos - 1] + self.l1
                    link_iter = 'l2'
                elif link_iter == 'l2':
                    self.chain[pos] = self.chain[pos - 1] + self.l2
                    link_iter = 'l1'
        self.contour_length = math.sqrt(np.dot(self.chain[-1] - self.chain[0], self.chain[-1] - self.chain[0]))

    def _random_angle(self):
        del self.dihedral_set[:]
        random = np.random.uniform(0.0, 1.0, size=(self.monomer_num - 1))
        angle_map = self.prob_angle[:, 0].searchsorted(random)
        for prob_i in angle_map:
            rand = np.random.uniform(0.0, 1.0)
            if rand < 0.5:
                self.dihedral_set.append(self.prob_angle[prob_i - 1][1])
                continue
            else:
                try:
                    self.dihedral_set.append(self.prob_angle[prob_i][1])
                except IndexError:
                    if prob_i == len(self.prob_angle):
                        self.dihedral_set.append(self.prob_angle[0][1])

    def rotate_chain(self):
        self._random_angle()
        self.relax_chain = np.array(self.chain, copy=True)
        ete = [0.0, self.monomer_len]
        pos_i = 1
        for angle in self.dihedral_set:
            uv = utils.unit_vector(self.relax_chain[pos_i], self.relax_chain[pos_i + 1])
            for pos_j in range(pos_i + 2, len(self.relax_chain), 1):
                self.relax_chain[pos_j] = \
                    utils.point_rotation(self.relax_chain[pos_j],
                                         angle, uv, origin=self.relax_chain[pos_i])
            pos_i += 2
            ete.append(math.sqrt(np.dot(
                self.relax_chain[pos_i] - self.relax_chain[0],
                self.relax_chain[pos_i] - self.relax_chain[0])))
        self.end_to_end = np.array(ete)

    def tangent_auto_corr(self, chain):
        tangent_vecs = np.array([chain[i + 1] - chain[i] for i in range(0, len(chain), 2)])
        tangent_uv = tangent_vecs / la.norm(tangent_vecs, axis=1)[0:, None]
        array_1 = np.copy(tangent_uv)
        array_2 = np.copy(array_1)
        for i in range(len(array_1)):
            corr = np.einsum('ij,ij->i', array_1, array_2)
            map(self.tangent_corr[i].update, corr)
            array_1 = np.delete(array_1, -1, 0)
            array_2 = np.delete(array_2, 0, 0)

    def _unit_normal_vectors(self, chain):
        vec1 = []
        vec2 = []
        for i in range(len(chain) - 2):
            if i == 0:
                vec1.append(chain[i + 1] - chain[i])
                vec2.append(chain[i + 2] - chain[i])
            if i % 2 != 0:
                vec1.append(chain[i + 1] - chain[i])
                vec2.append(chain[i + 2] - chain[i])
        normal = np.cross(vec2, vec1)
        self.unit_normal = normal / la.norm(normal, axis=1)[0:, None]

    def p2_order_param(self, chain=None, unit_vectors=None):
        if unit_vectors is None:
            self._unit_normal_vectors(chain)
            unit_vectors = self.unit_normal
        self.director_matrix = np.asmatrix((np.einsum('ij,ik->jk', (3 * unit_vectors), unit_vectors) *
                                            (1. / (2. * len(unit_vectors)))) - (np.identity(3) * (1. / 2.)))
        eigval, eigvect = la.eig(self.director_matrix)
        self.director = np.asarray(eigvect[:, np.argmax(eigval)]).flatten()
        for vec in unit_vectors:
            s = ((3./2.) * (np.dot(self.director, vec) ** 2)) - (1./2.)
            self.s_order_param.update(s)

    def p2_auto_corr(self, chain):
        self._unit_normal_vectors(chain)
        array_1 = np.copy(self.unit_normal)
        array_2 = np.copy(array_1)
        for i in range(len(array_1)):
            corr = ((np.einsum('ij,ij->i', array_1, array_2) ** 2) * (3. / 2.)) - (1. / 2.)
            map(self.s_x_corr[i].update, corr)
            array_1 = np.delete(array_1, -1, 0)
            array_2 = np.delete(array_2, 0, 0)

    def sample_chains(self):
        for chain_i in range(1, self.sample_num + 1, 1):
            self.rotate_chain()
            self.p2_order_param(self.relax_chain)
            self.ete_stats.update(self.end_to_end)
            self.dihedral_hist.update(self.dihedral_set)
            self.ete_hist.append(self.end_to_end[-1])


class RandomChargePolymer(Polymer):
    def __init__(self, monomer_num, monomer_length, link_length,
                 link_angle, prob_angle, c_monomer_length, c_link_length, c_link_angle, c_prob_angle, sample_num=None):
        super(RandomChargePolymer, self).__init__(monomer_num, monomer_length,
                                                  link_length, link_angle,
                                                  prob_angle, sample_num)
        self.c_prob_angle = c_prob_angle
        self.charged_chain = None
        self.c_end_to_end = None
        self.c_corr = None
        self.c_ete_stats = Stats()
        self.c_corr_stats = Stats()
        # set of dihedral angles for charged polymer
        self.c_dihedral_set = []
        # combined and shuffled list of dihedral angles
        self.shuffle_dihedral_set = []
        self.c_monomer_len = c_monomer_length
        self.c_link_len = c_link_length
        self.c_link_angle = (c_link_angle * np.pi / 180)
        self.c_indexes = []
        self.c_chain = np.zeros(((2 * monomer_num), 3))
        self.c_m = [self.c_monomer_len, 0, 0]
        self.c_l1 = [self.c_link_len * math.cos(self.c_link_angle), -self.c_link_len * math.sin(self.c_link_angle), 0]
        self.c_l2 = [self.c_link_len * math.cos(self.c_link_angle), self.c_link_len * math.sin(self.c_link_angle), 0]

    def _c_build_chain(self):
        # Loop over all dihedrals
        for i, d_angle in enumerate(self.shuffle_dihedral_set):
            link = None
            if i in self.c_indexes:
                monomer = self.c_m
                # cond statement to toggle between link 1 and link 2
                if i % 2 == 0:
                    link = self.c_l2
                else:
                    link = self.c_l1
            else:
                monomer = self.m
                if i % 2 == 0:
                    link = self.l2
                else:
                    link = self.l1
            # two atoms/beads added per dihedral
            pos_1 = (i * 2) + 1
            pos_2 = (i * 2) + 2
            # add monomer
            self.c_chain[pos_1] = self.c_chain[pos_1 - 1] + monomer
            # add link
            self.c_chain[pos_2] = self.c_chain[pos_2 - 1] + link
            # chain end
            if i == len(self.shuffle_dihedral_set) - 1:
                self.c_chain[pos_2 + 1] = self.c_chain[pos_2] + monomer
        self.contour_length = math.sqrt(np.dot(self.c_chain[-1] - self.c_chain[0], self.c_chain[-1] - self.c_chain[0]))

    def _c_random_angle(self, total_sites):
        del self.c_dihedral_set[:]  # clear dihedral set each time _c_random_angle is called
        random = np.random.uniform(0.0, 1.0, size=total_sites)
        angle_map = self.c_prob_angle[:, 0].searchsorted(random)
        for prob_i in angle_map:
            rand = np.random.uniform(0.0, 1.0)
            if rand < 0.5:
                self.c_dihedral_set.append(self.c_prob_angle[prob_i - 1][1])
            else:
                try:
                    self.c_dihedral_set.append(self.c_prob_angle[prob_i][1])
                except IndexError:
                    if prob_i == len(self.c_prob_angle):
                        self.c_dihedral_set.append(self.c_prob_angle[0][1])

    def shuffle_charged_chain(self, tot_charged_sites):
        self.shuffle_dihedral_set[:] = []
        self.c_indexes[:] = []
        self._c_random_angle(tot_charged_sites)
        self._random_angle()
        neutral_sites = (self.monomer_num - 1) - tot_charged_sites
        site_order = np.arange(0, self.monomer_num - 1, 1)
        np.random.shuffle(site_order)
        c_dihedral_list = self.dihedral_set[:neutral_sites] + self.c_dihedral_set
        for i, val in enumerate(site_order):
            self.shuffle_dihedral_set.append(c_dihedral_list[val])
            if val >= neutral_sites:
                self.c_indexes.append(i)
        # Finds the index of the charged dihedral angles
        self._c_build_chain()
        self.charged_chain = np.array(self.c_chain, copy=True)
        pos_i = 1
        for angle in self.shuffle_dihedral_set:
            uv = utils.unit_vector(self.charged_chain[pos_i], self.charged_chain[pos_i + 1])
            for pos_j in range(pos_i + 2, len(self.charged_chain), 1):
                self.charged_chain[pos_j] = utils.point_rotation(self.charged_chain[pos_j],
                                                                 angle, uv, origin=self.charged_chain[pos_i])
            pos_i += 2
        self.c_end_to_end = (math.sqrt(np.dot(
            self.charged_chain[-1] - self.charged_chain[0],
            self.charged_chain[-1] - self.charged_chain[0])))

    def sample_charged_chains(self, tot_charged_sites):
        for chain_i in range(1, self.sample_num + 1, 1):
            self.shuffle_charged_chain(tot_charged_sites)
            self.p2_order_param(self.charged_chain)
            self.c_ete_stats.update(self.c_end_to_end)
