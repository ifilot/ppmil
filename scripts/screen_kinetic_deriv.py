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
    atid = 1
    at = nuclei[atid][0]
    atchg = nuclei[atid][1]

    # geometric derivative for atom id
    geomatid = 1

    # load results from file
    fname = os.path.join(os.path.dirname(__file__), '../tests/data', 'nuclear_deriv_h2o.txt')
    #vals = np.loadtxt(fname).reshape((len(cgfs), len(cgfs), 3, 3))
    
    forces = np.zeros((len(cgfs), len(cgfs), 3))
    vals = np.zeros_like(forces)
    
    for i in range(0, len(cgfs)): # loop over cgfs
        for j in range(0, len(cgfs)): # loop over cgfs
            for l in range(0,3):  # loop over directions
                force = integrator.nuclear_deriv(cgfs[i], cgfs[j], at, atchg, nuclei[geomatid][0], l)
                val = calculate_force_finite_difference(molfile, basisfile, geomatid, i, j, l, atid)
                vals[i,j,l] = val
                forces[i,j,l] = force
                    
    diff = np.abs(forces - vals)
    
    cbarw = 0.5
    fig, ax = plt.subplots(3,3, dpi=144, figsize=(8,8))
    for i in range(0,3):
        for j in range(0,3):
            if i == 0:
                plot_matrix(ax[i,j], forces[:,:,j], 'Force [%s]' % (['x','y','z'])[j], cmap='BrBG')
            elif i == 1:
                plot_matrix(ax[i,j], diff[:,:,j], 'Error [%s]' % (['x','y','z'])[j], cmap='PiYG')
            else:
                plot_matrix(ax[i,j], vals[:,:,j], 'Finite difference [%s]' % (['x','y','z'])[j], cmap='BrBG')

    plt.tight_layout()
                        
def plot_matrix(ax, mat, title = None, cmap='PiYG'):
    """
    Produce plot of matrix
    """
    ax.imshow(mat, vmin=-1, vmax=1, cmap=cmap)
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            ax.text(i, j, '%.2f' % mat[j,i], ha='center', va='center',
                    fontsize=7)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.hlines(np.arange(1, mat.shape[0])-0.5, -0.5, mat.shape[0] - 0.5,
              color='black', linestyle='--', linewidth=1)
    ax.vlines(np.arange(1, mat.shape[0])-0.5, -0.5, mat.shape[0] - 0.5,
              color='black', linestyle='--', linewidth=1)
    
    if title:
        ax.set_title(title)

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