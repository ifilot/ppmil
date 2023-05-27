# -*- coding: utf-8 -*-

from pylebedev import PyLebedev
import numpy as np
from .gto import GTO

class Quadrature:
    def __init__(self):
        pass

    def quad_overlap(self, gto1, gto2, radial_points=32, lebedev_order=31):
        """
        Perform Gauss-Chebychev-Lebedev quadrature on trial wave function
        """        
        # verify that variables are GTOs
        if not isinstance(gto1, GTO):
            raise TypeError('Argument gto1 must be of GTO type')
        if not isinstance(gto2, GTO):
            raise TypeError('Argument gto2 must be of GTO type')

        # determine new product center
        rp = self.__gaussian_product_center(gto1.alpha, gto1.p, gto2.alpha, gto2.p)

        # get points
        gridw, gridpts = self.__construct_gcl_grid(radial_points, lebedev_order)
        gridpts += rp # position grid points at Gaussian product center
        
        # perform integration
        amsp1 = [gto1.get_amp(p) for p in gridpts]
        amsp2 = [gto2.get_amp(p) for p in gridpts]
        integral = np.sum(gridw * amsp1 * amsp2)
        
        return integral
    
    def quad_dipole(self, gto1, gto2, c, radial_points=32, lebedev_order=31):
        """
        Perform Gauss-Chebychev-Lebedev quadrature on trial wave function
        """
        # verify that variables are GTOs
        if not isinstance(gto1, GTO):
            raise TypeError('Argument gto1 must be of GTO type')
        if not isinstance(gto2, GTO):
            raise TypeError('Argument gto2 must be of GTO type')

        # get points
        gridw, gridpts = self.__construct_gcl_grid(radial_points, lebedev_order)
        
        # perform integration
        amsp1 = [gto1.get_amp(p) for p in gridpts]
        amsp2 = [gto2.get_amp(p) for p in gridpts]
        
        potx = [p[0] for p in gridpts]
        poty = [p[1] for p in gridpts]
        potz = [p[2] for p in gridpts]
        
        integralx = np.sum(gridw * amsp1 * potx * amsp2)
        integraly = np.sum(gridw * amsp1 * poty * amsp2)
        integralz = np.sum(gridw * amsp1 * potz * amsp2)
        
        return np.array([integralx, integraly, integralz])

    def __construct_gcl_grid(self, radial_points=32, lebedev_order=31, rm=1.0):
        """
        Perform Gauss-Chebychev-Lebedev quadrature on trial wave function
        """
        # create grid points
        N = radial_points   # number of grid points

        # build the Gauss-Chebychev grid following the canonical recipe
        z = np.arange(1, N+1)
        x = np.cos(np.pi / (N+1) * z)
        r = rm * (1 + x) / (1 - x)
        wr = np.pi / (N+1) * np.sin(np.pi / (N+1) * z)**2 * 2.0 * rm \
            / (np.sqrt(1 - x**2) * (1 - x)**2)

        # get Lebedev points
        leblib = PyLebedev()
        p,wl = leblib.get_points_and_weights(lebedev_order)

        # construct full grid
        gridpts = np.outer(r, p).reshape((-1,3))
        gridw = np.outer(wr * r**2, wl).flatten() * 4.0 * np.pi

        return gridw, gridpts

    def __gaussian_product_center(self, alpha1, a, alpha2, b):
        """
        Calculate the position of the product of two Gaussians
        """
        return (alpha1 * a + alpha2 * b) / (alpha1 + alpha2)