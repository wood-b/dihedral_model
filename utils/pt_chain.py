import numpy as np
from pymatgen import Molecule
import dihedral_tools
from numpy.linalg import norm
import utils


class PtPolymer(object):
    def __init__(self, start_monomer, s_head, s_tail,
                 monomer, head, tail,
                 end_monomer, e_head, e_tail,
                 n_units, link_distance, link_angle):
        self.n_units = n_units
        self.link_distance = link_distance
        self.link_angle = (link_angle * np.pi / 180.0)
        self.ordered_chain = None
        self.rotated_chain = None
        self.dihedral_atoms = None

        start_monomer = self.tangent_along_x(start_monomer, s_head, s_tail)
        start_monomer = self.align_ring_normal_along_z(start_monomer)
        self.start_monomer, self.s_head, self.s_tail = self.order_atoms(start_monomer, s_head, s_tail)

        monomer = self.tangent_along_x(monomer, head, tail)
        monomer = self.align_ring_normal_along_z(monomer)
        self.monomer, self.head, self.tail = self.order_atoms(monomer, head, tail)

        end_monomer = self.tangent_along_x(end_monomer, e_head, e_tail)
        end_monomer = self.align_ring_normal_along_z(end_monomer)
        self.end_monomer, self.e_head, self.e_tail = self.order_atoms(end_monomer, e_head, e_tail)

        # rotate bulk monomer to create trans chain
        self.trans_monomer = self.monomer.copy()
        self.trans_monomer.rotate_sites(range(len(self.trans_monomer)), theta=np.pi, axis=[0, 0, 1], anchor=[0, 0, 0])
        self.trans_monomer.translate_sites(range(len(self.trans_monomer)), - self.trans_monomer.cart_coords[self.tail])
        self.trans_monomer, self.trans_head, self.trans_tail = self.order_atoms(self.trans_monomer,
                                                                                self.tail, self.head)

        '''self.start_monomer.translate_sites(range(len(start_monomer)),
                                           - start_monomer.cart_coords[s_head])
        self.monomer.translate_sites(range(len(monomer)),
                                     - monomer.cart_coords[head])
        self.rot_monomer = self.monomer.copy()
        self.rot_monomer.rotate_sites(range(len(self.rot_monomer)),
                                      theta=np.pi, axis=[0,0,1], anchor=[0,0,0])
        self.rot_monomer.translate_sites(range(len(monomer)),
                                     - self.monomer.cart_coords[tail])
        self.end_monomer.translate_sites(range(len(end_monomer)),
                                         - end_monomer.cart_coords[e_head])'''
        self.chain = self.start_monomer.copy()
        self.l1 = np.array([self.link_distance * np.cos(self.link_angle),
                            -self.link_distance * np.sin(self.link_angle), 0])
        self.l2 = np.array([self.link_distance * np.cos(self.link_angle),
                            self.link_distance * np.sin(self.link_angle), 0])
        self._build()
        #self._order()

    # tools for aligning monomers
    def tangent_along_x(self, mol, head_atom, tail_atom):
        mol.translate_sites(range(len(mol)), - mol.cart_coords[head_atom])
        if mol.cart_coords[tail_atom][0] < 0:
            unit_vec = utils.unit_vector(mol.cart_coords[tail_atom], mol.cart_coords[head_atom])
        if mol.cart_coords[tail_atom][0] > 0:
            unit_vec = utils.unit_vector(mol.cart_coords[head_atom], mol.cart_coords[tail_atom])
        rot_axis = np.cross(unit_vec, [1, 0, 0]) / norm(np.cross(unit_vec, [1, 0, 0]))
        deg_to_rot = np.arccos(np.dot(unit_vec, [1, 0, 0])) * (180.0 / np.pi)
        coords = [utils.point_rotation(mol.cart_coords[i], deg_to_rot, rot_axis, origin=[0, 0, 0])
                  for i in range(len(mol))]
        return Molecule(mol.species, coords)

    def align_ring_normal_along_z(self, mol):
        mol_dict = mol.as_dict()
        norm_pts = []
        for i in mol_dict['sites']:
            if i['name'] != 'H':
                norm_pts.append(np.array([i['xyz'][0], i['xyz'][1], i['xyz'][2]]))
        vec_1 = norm_pts[2] - norm_pts[1]
        vec_2 = norm_pts[3] - norm_pts[1]
        unit_normal = np.cross(vec_1, vec_2) / norm(np.cross(vec_1, vec_2))
        theta = np.arccos(np.dot(unit_normal, np.array([0, 0, 1])))
        rot_axis = np.cross(unit_normal, [0, 0, 1]) / norm(np.cross(unit_normal, [0, 0, 1]))
        axis = None
        if rot_axis[0] < 0:
            axis = np.array([-1, 0, 0])
        else:
            axis = np.array([1, 0, 0])
        mol.rotate_sites(range(len(mol)), theta=theta, axis=axis, anchor=[0, 0, 0])
        new_mol_dict = mol.as_dict()
        for i in new_mol_dict['sites']:
            if i['name'] == 'S':
                s_y_coord = i['xyz'][1]
        if s_y_coord < 0:
            mol.rotate_sites(range(len(mol)), theta=np.pi, axis=axis, anchor=[0, 0, 0])
        return mol

    def order_atoms(self, mol, head_atom, tail_atom):
        unsorted = mol.as_dict()
        x_list = []
        full_list = []
        for i in unsorted['sites']:
            full_list.append([i['xyz'][0], i['xyz'][1], i['xyz'][2], i['name']])
            x_list.append(i['xyz'][0])
        x_sorted = sorted(x_list)
        coords = []
        species = []
        for x_val in x_sorted:
            i = x_list.index(x_val)
            coords.append([full_list[i][0], full_list[i][1], full_list[i][2]])
            species.append(full_list[i][3])
        head_i = unsorted['sites'][head_atom]['xyz'][0]
        head_new = x_sorted.index(head_i)
        tail_i = unsorted['sites'][tail_atom]['xyz'][0]
        tail_new = x_sorted.index(tail_i)
        return Molecule(species, coords), head_new, tail_new

    # build is only intended to work with structures aligned on the positive x-axis
    def _build(self):
        i_tail = self.s_tail
        for i in range(self.n_units - 1):
            link = None
            add_monomer = None
            if i % 2 != 0:
                link = self.l1
                add_monomer = self.monomer.copy()
            if i % 2 == 0:
                link = self.l2
                add_monomer = self.trans_monomer.copy()
            if i == self.n_units - 2:
                add_monomer = self.end_monomer.copy()
                add_monomer.translate_sites(range(len(self.end_monomer)), (self.chain.cart_coords[i_tail] + link))
            else:
                add_monomer.translate_sites(range(len(self.monomer)), (self.chain.cart_coords[i_tail] + link))
            for j, site in enumerate(add_monomer):
                self.chain.append(site.specie, site.coords)
            if i == 0:
                i_tail = len(self.start_monomer) + self.tail
            else:
                i_tail += len(self.monomer)

    def _order(self):
        unsorted = self.chain.as_dict()
        x_list = []
        full_list = []
        for i in unsorted['sites']:
            full_list.append([i['xyz'][0], i['xyz'][1], i['xyz'][2], i['name']])
            x_list.append(i['xyz'][0])
        x_sorted = sorted(x_list)
        coords = []
        species = []
        for x_val in x_sorted:
            i = x_list.index(x_val)
            coords.append([full_list[i][0], full_list[i][1], full_list[i][2]])
            species.append(full_list[i][3])
        self.ordered_chain = Molecule(species, coords)

    def _find_dihedrals(self):
        unsorted = self.chain.as_dict()
        x_list = []
        full_list = []
        for i in unsorted['sites']:
            full_list.append([i['xyz'][0], i['xyz'][1], i['xyz'][2], i['name']])
            x_list.append(i['xyz'][0])
        x_sorted = sorted(x_list)
        coords = []
        species = []
        for x_val in x_sorted:
            i = x_list.index(x_val)
            coords.append([full_list[i][0], full_list[i][1], full_list[i][2]])
            species.append(full_list[i][3])
        self.rotated_chain = Molecule(species, coords)
        self.dihedral_atoms = []
        for i, val in enumerate(self.rotated_chain.species):
            if str(val) == "C" and i > 3 and i < (len(self.rotated_chain.species) - 5):
                self.dihedral_atoms.append(i + 1)
        self.dihedral_atoms = np.array(self.dihedral_atoms).reshape((self.n_units - 1, 4))

    def spin_dihedrals(self, dihedral_angle):
        self._find_dihedrals()
        for i, val in enumerate(self.dihedral_atoms):
            self.rotated_chain = dihedral_tools.set_dihedral(self.rotated_chain, val,
                                                             dihedral_angle[i], write_xyz=False)

    def write_xyz(self, filename):
        self.rotated_chain.to(filename=filename, fmt="xyz")