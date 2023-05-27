# -*- coding: utf-8 -*-

import json
import numpy as np
from .cgf import CGF
from .gto import GTO

class Molecule:
    """
    Molecule class
    """
    def __init__(self, _name='unknown', xyzfile=None):
        self.atoms = []
        self.charges = []
        self.name = _name
        
        if xyzfile is not None:
            self.__load_from_xyz(xyzfile)

    def __str__(self):
        res = "Molecule: %s\n" % self.name
        for atom in self.atoms:
            res += " %s (%f,%f,%f)\n" % (atom[0], atom[1][0], atom[1][1], atom[1][2])

        return res

    def add_atom(self, atom, x, y, z, unit='bohr'):
        ang2bohr = 1.88973

        if unit == "bohr":
            self.atoms.append([atom, np.array([x, y, z])])
        elif unit == "angstrom":
            self.atoms.append([atom, np.array([x*ang2bohr, y*ang2bohr, z*ang2bohr])])
        else:
            raise RuntimeError("Invalid unit encountered: %s. Accepted units are 'bohr' and 'angstrom'." % unit)

        self.charges.append(0)

    def build_basis(self, name, basis_filename):
        f = open(basis_filename, 'r')
        basis = json.load(f)
        f.close()
        
        self.cgfs = []
        self.nuclei = []

        for aidx, atom in enumerate(self.atoms):
            cgfs_template = basis[atom[0]]

            # store information about the nuclei
            self.charges[aidx] = cgfs_template['atomic_number']
            self.nuclei.append([atom[1], cgfs_template['atomic_number']])

            for cgf_t in cgfs_template['cgfs']:
                # s-orbitals
                if cgf_t['type'] == 'S':
                    self.cgfs.append(CGF(atom[1]))
                    for gto in cgf_t['gtos']:
                        self.cgfs[-1].add_gto(gto['coeff'], gto['alpha'], 0, 0, 0)

                # p-orbitals
                if cgf_t['type'] == 'P':
                    self.cgfs.append(CGF(atom[1]))
                    for gto in cgf_t['gtos']:
                        self.cgfs[-1].add_gto(gto['coeff'], gto['alpha'], 1, 0, 0)
                    
                    self.cgfs.append(CGF(atom[1]))
                    for gto in cgf_t['gtos']:
                        self.cgfs[-1].add_gto(gto['coeff'], gto['alpha'], 0, 1, 0)
                    
                    self.cgfs.append(CGF(atom[1]))
                    for gto in cgf_t['gtos']:
                        self.cgfs[-1].add_gto(gto['coeff'], gto['alpha'], 0, 0, 1)

                # d-orbitals
                if cgf_t['type'] == 'D':
                    self.cgfs.append(CGF(atom[1]))
                    for gto in cgf_t['gtos']:
                        self.cgfs[-1].add_gto(gto['coeff'], gto['alpha'], 2, 0, 0)
                    
                    self.cgfs.append(CGF(atom[1]))
                    for gto in cgf_t['gtos']:
                        self.cgfs[-1].add_gto(gto['coeff'], gto['alpha'], 0, 2, 0)
                    
                    self.cgfs.append(CGF(atom[1]))
                    for gto in cgf_t['gtos']:
                        self.cgfs[-1].add_gto(gto['coeff'], gto['alpha'], 0, 0, 2)
                    
                    self.cgfs.append(CGF(atom[1]))
                    for gto in cgf_t['gtos']:
                        self.cgfs[-1].add_gto(gto['coeff'], gto['alpha'], 1, 1, 0)
                    
                    self.cgfs.append(CGF(atom[1]))
                    for gto in cgf_t['gtos']:
                        self.cgfs[-1].add_gto(gto['coeff'], gto['alpha'], 1, 0, 1)
                    
                    self.cgfs.append(CGF(atom[1]))
                    for gto in cgf_t['gtos']:
                        self.cgfs[-1].add_gto(gto['coeff'], gto['alpha'], 0, 1, 1)

        return self.cgfs, self.nuclei
    
    def __load_from_xyz(self, xyzfile):
        f = open(xyzfile, 'r')
        nratoms = int(f.readline().strip())
        f.readline() # skip line
        for i in range(nratoms):
            pieces = f.readline().strip().split()
            atom = str(pieces[0])
            x = float(pieces[1])
            y = float(pieces[2])
            z = float(pieces[3])
        
            self.add_atom(atom, x, y, z, unit='angstrom')