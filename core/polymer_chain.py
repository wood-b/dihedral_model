import numpy as np
import math
import utils

__author__ = "Brandon Wood"


class Polymer(object):
    def __init__(self, monomer_num, monomer_length, link_length,
                 link_angle, n_prob_angle, sample_num=None, c_prob_angle=None):
        """
        :param monomer_num: number of monomers in polymer chain
        :param monomer_length: length of monomer tangent line
        :param link_length: length of link between monomers
        :param link_angle: angle between monomer tangent and link
        :param n_prob_angle: array of dihedral probability and angle pairs (neutral case)
        :param sample_num: number of chains to sample
        :param c_prob_angle: array of dihedral probability and angle pairs (charged case)
        """
        self.chain = np.zeros(((2 * monomer_num), 3))
        self.relax_chain = np.zeros(((2 * monomer_num), 3))
        self.monomer_num = monomer_num
        self.monomer_len = monomer_length
        self.link_len = link_length
        self.link_angle = (link_angle * np.pi / 180)
        self.n_prob_angle = n_prob_angle
        self.sample_num = sample_num
        self.c_prob_angle = c_prob_angle
        self.end_to_end = None
        self.c_ete = None
        self.corr = None
        self.c_corr = None
        # position monomer tangent and links
        self.m = [self.monomer_len, 0, 0]
        self.l1 = [self.link_len * math.cos(self.link_angle),
                   -self.link_len * math.sin(self.link_angle), 0]
        self.l2 = [self.link_len * math.cos(self.link_angle),
                   self.link_len * math.sin(self.link_angle), 0]

    def build_chain(self, chain_return=False):
        link_iter = 'l1'
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
        if chain_return:
            return self.chain

    def n_random_angle(self):
        random = np.random.uniform(0.0, 1.0, size=(self.monomer_num - 1))
        angle_map = self.n_prob_angle[:, 0].searchsorted(random)
        dihedral_set = []
        for prob_i in angle_map:
            rand = np.random.uniform(0.0, 1.0)
            if rand < 0.5:
                dihedral_set.append(self.n_prob_angle[prob_i - 1][1])
                continue
            else:
                try:
                    dihedral_set.append(self.n_prob_angle[prob_i][1])
                except IndexError:
                    if prob_i == 3600:
                        dihedral_set.append(self.n_prob_angle[0][1])
        return dihedral_set

    def relax_neutral_chain(self, values=False):
        dihedral_set = self.n_random_angle()
        self.relax_chain = np.array(self.chain, copy=True)
        ete = [0.0, self.monomer_len]
        pos_i = 1
        for angle in dihedral_set:
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
        self.corr = np.array([utils.correlation(
            self.relax_chain[0], self.relax_chain[1],
            self.relax_chain[i], self.relax_chain[i+1])
                                 for i in range(0, len(self.chain), 2)])
        if values:
            return self.relax_chain, self.end_to_end

    def sample_neutral_chains(self):
        ave_ete = np.zeros(self.monomer_num + 1)
        ave_corr = np.zeros(self.monomer_num)
        self.build_chain()
        for chain_i in range(self.sample_num):
            self.relax_neutral_chain()
            ave_ete += self.end_to_end
            ave_corr += self.corr
        ave_ete /= self.sample_num
        ave_corr /= self.sample_num
        return ave_ete, ave_corr

    def c_random_angle(self, sites):
        random = np.random.uniform(0.0, 1.0, size=sites)
        angle_map = self.c_prob_angle[:, 0].searchsorted(random)
        dihedral_set = []
        for prob_i in angle_map:
            rand = np.random.uniform(0.0, 1.0)
            if rand < 0.5:
                dihedral_set.append(self.c_prob_angle[prob_i - 1][1])
            else:
                try:
                    dihedral_set.append(self.c_prob_angle[prob_i][1])
                except IndexError:
                    if prob_i == 3600:
                        dihedral_set.append(self.c_prob_angle[0][1])
        return dihedral_set

    def pick_links(self, sites):
        link_pos = []
        excluded = []
        while len(link_pos) < sites:
            # electron can't be located 5 units from either chain end
            # chain needs to be at least 10 units long
            test_link = np.random.randint(5, (self.monomer_num - 4))
            if test_link not in excluded:
                link_pos.append(test_link)
                excluded.extend([test_link - 1, test_link, test_link + 1])
        return sorted(link_pos)

    def relax_charged_chain(self, electrons, values=False):
        self.relax_neutral_chain()
        # electron affects dihedral angle of 3 links
        sites = int(math.ceil(electrons / 2.0)) * 3
        dihedral_set = self.c_random_angle(sites)
        charged_chain = np.array(self.relax_chain, copy=True)
        pick_links = zip(self.pick_links(sites), dihedral_set)
        for link, angle in pick_links:
            # idx is the bead or atom index on the chain
            idx = (link * 2) - 1
            uv = utils.unit_vector(charged_chain[idx], charged_chain[idx + 1])
            for pos_j in range(idx + 2, len(charged_chain), 1):
                charged_chain[pos_j] = utils.point_rotation(charged_chain[pos_j], angle, uv, origin=charged_chain[idx])
        # sampling
        self.c_ete = (math.sqrt(np.dot(
            charged_chain[len(charged_chain)-1] - charged_chain[0],
            charged_chain[len(charged_chain)-1] - charged_chain[0])))
        self.c_corr = np.array([utils.correlation(
            charged_chain[0], charged_chain[1],
            charged_chain[i], charged_chain[i+1])
                                 for i in range(0, len(self.chain), 2)])
        if values:
            return self.relax_chain, self.end_to_end, charged_chain, self.c_ete

    def sample_charged_chains(self, electrons):
        ave_ete = 0.0
        ave_corr = np.zeros(self.monomer_num)
        self.build_chain()
        for chain_i in range(self.sample_num):
            self.relax_charged_chain(electrons)
            ave_ete += self.c_ete
            ave_corr += self.c_corr
        ave_ete /= self.sample_num
        ave_corr /= self.sample_num
        return ave_ete, ave_corr
