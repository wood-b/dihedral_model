# This code is compatible with RandomChargedPolymer Class but is not functioning properly.
# Value of order param is too low something might be wrong with the chain rotation???

def _pick_links(self, sites, polaron_size):
    # links are indexed from 1
    total_sites = self.monomer_num - 1
    neutral_sites = total_sites - (sites * polaron_size)
    num_neutral_sites = []
    for i in range(1000):
        neutral_spacing = np.random.uniform(size=(sites + 1)) * sites * 10
        num_neutral_sites = np.rint(neutral_spacing * (neutral_sites / sum(neutral_spacing)))
        if sum(num_neutral_sites) == neutral_sites:
            break
    if sum(num_neutral_sites) != neutral_sites:
        print "picking links error"
    links = []
    # places the charged links
    for i in range(len(num_neutral_sites) - 1):
        if i == 0:
            links.extend(np.arange(1, (polaron_size + 1)) + num_neutral_sites[i])
        else:
            links.extend(np.arange(1, (polaron_size + 1)) + (links[-1] + num_neutral_sites[i]))
    self.c_links = links


def rotate_charged_chain(self, percent_excited, polaron_size):
    self.rotate_chain()
    # percent_excited is the desired percentage of dihedral angles impacted by excitation
    # polaron_size is number of sequential dihedral angles affected by an excitation
    sites = int(math.floor(((percent_excited / 100.) * (self.monomer_num - 1.)) / float(polaron_size)))
    if sites == 0:
        print "percent excited is too low"
    if sites != 0:
        self.actual_percent_excited = ((sites * polaron_size) / (self.monomer_num - 1.) * 100.)
        self._c_random_angle(sites * polaron_size)
        self.charged_chain = np.array(self.relax_chain, copy=True)
        self._pick_links(sites, polaron_size)
        pick_links = zip(self.c_links, self.c_dihedral_set)
        for link, angle in pick_links:
            # idx is the bead or atom index on the chain
            idx = int((link * 2.) - 1.)
            uv = utils.unit_vector(self.charged_chain[idx], self.charged_chain[idx + 1])
            for pos_j in range(idx + 2, len(self.charged_chain), 1):
                self.charged_chain[pos_j] = utils.point_rotation(self.charged_chain[pos_j],
                                                                 angle, uv, origin=self.charged_chain[idx])
    # sampling
    self.c_end_to_end = (math.sqrt(np.dot(
        self.charged_chain[-1] - self.charged_chain[0],
        self.charged_chain[-1] - self.charged_chain[0])))
