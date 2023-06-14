import unittest
from copy import copy, deepcopy
import numpy as np
import os, sys
import matplotlib.pyplot as plt

# add a reference to load the PPMIL library
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from ppmil import Molecule, PPMIL
  
def main():
    """
    Test Derivatives of water
    """
    # build integrator object
    integrator = PPMIL()

    # build hydrogen molecule
    molfile = os.path.join(os.path.dirname(__file__), '../tests/data', 'h2o.xyz')
    mol = Molecule(xyzfile=molfile)
    basisfile = os.path.join(os.path.dirname(__file__), '../tests/data', 'sto3g.json')
    cgfs, nuclei = mol.build_basis('sto3g', basisfile)
    
    # electronic interaction with atom atid
    atid = 0
    at = nuclei[atid][0]
    atchg = nuclei[atid][1]

    # load results from file
    fname = os.path.join(os.path.dirname(__file__), '../tests/data', 'nuclear_deriv_h2o.txt')
    #vals = np.loadtxt(fname).reshape((len(cgfs), len(cgfs), 3, 3))
    
    forces = np.zeros((len(cgfs), len(cgfs), 3, 3))
    vals = np.zeros_like(forces)
    
    for i in range(0, len(cgfs)): # loop over cgfs
        for j in range(0, len(cgfs)): # loop over cgfs
            for k in range(0,3):  # loop over nuclei
                for l in range(0,3):  # loop over directions
                    force = integrator.nuclear_deriv(cgfs[i], cgfs[j], at, atchg, nuclei[k][0], l)
                    val = calculate_force_finite_difference(molfile, basisfile, k, i, j, l, atid)
                    vals[i,j,k,l] = val
                    forces[i,j,k,l] = force

    print(vals[4,5,0,2])
    print(vals[5,4,0,2])
    
    print(forces[4,5,0,2])
    print(forces[5,4,0,2])
                    
    diff = np.abs(forces - vals)
    
    fig, ax = plt.subplots(2,3, dpi=144)
    for i in range(0,2):
        for j in range(0,3):
            if i == 0:
                ax[i,j].imshow(diff[:,:,0,j], vmin=0, vmax=0.1)
                ax[i,j].set_title('Error [%s]' % (['x','y','z'])[j])
            else:
                ax[i,j].imshow(vals[:,:,0,j])
                ax[i,j].set_title('Force [%s]' % (['x','y','z'])[j])

    plt.tight_layout()
                        

def calculate_force_finite_difference(molfile, basisfile, 
                                      nuc_id, cgf_id1, cgf_id2, coord,
                                      atid):
    # build integrator object
    integrator = PPMIL()

    # distance
    diff = 0.00001

    vals = np.zeros(2)
    for i,v in enumerate([-1,1]):
        mol = Molecule(xyzfile=molfile)
        mol.atoms[nuc_id][1][coord] += v * diff / 2
        cgfs, nuclei = mol.build_basis('sto3g', basisfile)
        at = nuclei[atid][0]
        atchg = nuclei[atid][1]
        
        vals[i] = integrator.nuclear(cgfs[cgf_id1], cgfs[cgf_id2], at, atchg)

    return (vals[1] - vals[0]) / diff

if __name__ == '__main__':
    main()