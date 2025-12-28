import numpy as np
from scipy.special import factorial

from ..util.gto import GTO
from ..util.cgf import CGF
from .nuclear_engine import NuclearEngine
from ..math.math import binomial_prefactor, gaussian_product_center
from ..math.gamma import Fgamma

class HuzinagaNuclearEngine(NuclearEngine):

    def nuclear_primitive(self, gto1:GTO, gto2:GTO, nucleus):
        """
        Calculate nuclear attraction integral for two GTOs
        """
        return self._nuclear(gto1.p, gto1.o, gto1.alpha,
                              gto2.p, gto2.o, gto2.alpha,
                              nucleus)

    def _nuclear(self, a, o1, alpha1, b, o2, alpha2, c):
        """
        Calculate nuclear term
        
        a:      position of gto1
        o1:     orders of gto1
        alpha1: exponent of gto1
        b:      position of gto2
        o2:     orders of gto2
        alpha2: exponent of gto2
        c:      position of nucleus
        """
        gamma = alpha1 + alpha2
        p = gaussian_product_center(alpha1, a, alpha2, b)
        rab2 = np.sum(np.power(a-b,2))
        rcp2 = np.sum(np.power(c-p,2))
        
        ax = self._A_array(o1[0], o2[0], p[0] - a[0], p[0] - b[0], p[0] - c[0], gamma)
        ay = self._A_array(o1[1], o2[1], p[1] - a[1], p[1] - b[1], p[1] - c[1], gamma)
        az = self._A_array(o1[2], o2[2], p[2] - a[2], p[2] - b[2], p[2] - c[2], gamma)
        
        # pre-calculate Fgamma values
        nu_max = np.sum(o1) + np.sum(o2)
        fg = np.array([Fgamma(nu, gamma*rcp2) for nu in range(nu_max+1)])

        s = 0.0
        for i in range(o1[0] + o2[0] + 1):
            for j in range(o1[1] + o2[1] + 1):
                for k in range(o1[2] + o2[2] + 1):
                    s += ax[i] * ay[j] * az[k] * fg[i+j+k]
       
        return -2.0 * np.pi / gamma * np.exp(-alpha1*alpha2*rab2/gamma) * s
    
    def _A_array(self, l1, l2, pa, pb, cp, g):
        imax = l1 + l2 +1
        arr = np.zeros(imax)
        
        for i in range(imax):
            for r in range(i//2+1):
                for u in range((i-2*r)//2+1):
                    iI = i - 2*r - u
                    arr[iI] += self._A_term(i, r, u, l1, l2, pa, pb, cp, g)
        
        return arr
                    
    def _A_term(self, i, r, u, l1, l2, pax, pbx, cpx, gamma):
        return (-1)**i * binomial_prefactor(i, l1, l2, pax, pbx) * \
               (-1)**u * factorial(i) * np.power(cpx,i - 2*r - 2*u) * \
               np.power(0.25/gamma,r+u) / factorial(r) / factorial(u) / \
               factorial(i - 2*r - 2*u)
    
    def nuclear_deriv(self, cgf1, cgf2, nuc, charge, nucderiv, coord):
        """
        Analytic geometric derivative of the nuclear attraction integral, using
        the SAME displacement semantics as your finite-difference version:

        When differentiating w.r.t. a nuclear coordinate, we translate:
        - the nucleus itself if nucderiv == nuc
        - cgf1 center if it sits on nucderiv
        - cgf2 center if it sits on nucderiv

        The resulting derivative is the sum of the corresponding partial derivatives
        (A, B, and/or C contributions). If all three (cgf1, cgf2, nuc) coincide with
        nucderiv (pure translation), derivative is 0.
        """
        if not isinstance(cgf1, CGF):
            raise TypeError("Argument cgf1 must be of CGF type")
        if not isinstance(cgf2, CGF):
            raise TypeError("Argument cgf2 must be of CGF type")
        nuc = np.asarray(nuc, dtype=float)
        nucderiv = np.asarray(nucderiv, dtype=float)
        assert nuc.shape == (3,)
        assert nucderiv.shape == (3,)
        assert coord in (0, 1, 2)

        # Match your old tolerance scale (you used 1e-3)
        tol = 1e-3
        n1 = np.linalg.norm(cgf1.p - nucderiv) < tol
        n2 = np.linalg.norm(cgf2.p - nucderiv) < tol
        n3 = np.linalg.norm(nuc - nucderiv) < tol

        # Preserve your "pure translation" early exit behavior:
        # - all True  => moving cgf1, cgf2, and nucleus together => 0 by invariance
        # - all False => you're not moving anything relevant => 0
        if n1 == n2 == n3:
            return 0.0

        # Helper: evaluate primitive nuclear attraction with modified angular momenta.
        def prim_with_orders(g1, g2, o1_new, o2_new):
            return self._nuclear(
                g1.p, np.asarray(o1_new, dtype=int), g1.alpha,
                g2.p, np.asarray(o2_new, dtype=int), g2.alpha,
                nuc
            )

        dval = 0.0

        # ---- Contribution from moving CGF1 center (A) ----
        if n1:
            for gto1 in cgf1.gtos:
                for gto2 in cgf2.gtos:
                    l = int(gto1.o[coord])
                    aexp = float(gto1.alpha)

                    o1_up = np.array(gto1.o, dtype=int); o1_up[coord] += 1
                    v_up = prim_with_orders(gto1, gto2, o1_up, gto2.o)

                    if l > 0:
                        o1_dn = np.array(gto1.o, dtype=int); o1_dn[coord] -= 1
                        v_dn = prim_with_orders(gto1, gto2, o1_dn, gto2.o)
                    else:
                        v_dn = 0.0

                    # ∂/∂A_coord V = 2*alpha1*V(l+1) - l*V(l-1)
                    dv = 2.0 * aexp * v_up - float(l) * v_dn

                    pref = gto1.c * gto2.c * gto1.norm * gto2.norm
                    dval += pref * dv

        # ---- Contribution from moving CGF2 center (B) ----
        if n2:
            for gto1 in cgf1.gtos:
                for gto2 in cgf2.gtos:
                    l = int(gto2.o[coord])
                    bexp = float(gto2.alpha)

                    o2_up = np.array(gto2.o, dtype=int); o2_up[coord] += 1
                    v_up = prim_with_orders(gto1, gto2, gto1.o, o2_up)

                    if l > 0:
                        o2_dn = np.array(gto2.o, dtype=int); o2_dn[coord] -= 1
                        v_dn = prim_with_orders(gto1, gto2, gto1.o, o2_dn)
                    else:
                        v_dn = 0.0

                    # ∂/∂B_coord V = 2*alpha2*V(l+1) - l*V(l-1)
                    dv = 2.0 * bexp * v_up - float(l) * v_dn

                    pref = gto1.c * gto2.c * gto1.norm * gto2.norm
                    dval += pref * dv

        # ---- Contribution from moving the nucleus (C) ----
        if n3:
            for gto1 in cgf1.gtos:
                for gto2 in cgf2.gtos:
                    grad_c = self._nuclear_grad_c(
                        gto1.p, gto1.o, gto1.alpha,
                        gto2.p, gto2.o, gto2.alpha,
                        nuc
                    )
                    pref = gto1.c * gto2.c * gto1.norm * gto2.norm
                    dval += pref * grad_c[coord]

        return float(charge) * dval

    def _nuclear_grad_c(self, a, o1, alpha1, b, o2, alpha2, c):
        """
        Analytic gradient of the primitive nuclear attraction integral w.r.t. nuclear center c.
        Returns a length-3 numpy array: [dV/dcx, dV/dcy, dV/dcz].

        This matches your _nuclear() definition exactly.
        """
        gamma = alpha1 + alpha2
        p = gaussian_product_center(alpha1, a, alpha2, b)

        rab2 = np.sum((a - b) ** 2)
        cp = p - c
        rcp2 = np.sum(cp ** 2)

        # A arrays
        ax = self._A_array(o1[0], o2[0], p[0] - a[0], p[0] - b[0], p[0] - c[0], gamma)
        ay = self._A_array(o1[1], o2[1], p[1] - a[1], p[1] - b[1], p[1] - c[1], gamma)
        az = self._A_array(o1[2], o2[2], p[2] - a[2], p[2] - b[2], p[2] - c[2], gamma)

        # Boys values need up to nu_max+1 because d/dc introduces F_{n+1}
        nu_max = int(np.sum(o1) + np.sum(o2))
        fg = np.array([Fgamma(nu, gamma * rcp2) for nu in range(nu_max + 2)])

        # Prefactor is independent of c
        pref = -2.0 * np.pi / gamma * np.exp(-alpha1 * alpha2 * rab2 / gamma)

        # Helper limits
        nx = o1[0] + o2[0] + 1
        ny = o1[1] + o2[1] + 1
        nz = o1[2] + o2[2] + 1

        grad = np.zeros(3)

        # For each coordinate d, only A_d has explicit c-dependence through cp_d,
        # but Boys depends on all cp components via T = gamma*|cp|^2.
        for d in range(3):
            if d == 0:
                dax = self._A_array_dcd(o1[0], o2[0], p[0] - a[0], p[0] - b[0], p[0] - c[0], gamma)
                day = None
                daz = None
                cp_d = cp[0]
            elif d == 1:
                day = self._A_array_dcd(o1[1], o2[1], p[1] - a[1], p[1] - b[1], p[1] - c[1], gamma)
                dax = None
                daz = None
                cp_d = cp[1]
            else:
                daz = self._A_array_dcd(o1[2], o2[2], p[2] - a[2], p[2] - b[2], p[2] - c[2], gamma)
                dax = None
                day = None
                cp_d = cp[2]

            ds = 0.0
            # ∂F_n/∂c_d = 2*gamma*cp_d*F_{n+1}(T)
            # because dT/dc_d = -2*gamma*cp_d and dF_n/dT = -F_{n+1}
            boys_factor = 2.0 * gamma * cp_d

            for i in range(nx):
                Ai = ax[i]
                dAi = (dax[i] if dax is not None else 0.0)
                for j in range(ny):
                    Aj = ay[j]
                    dAj = (day[j] if day is not None else 0.0)
                    for k in range(nz):
                        Ak = az[k]
                        dAk = (daz[k] if daz is not None else 0.0)

                        n = i + j + k

                        # term from derivative of A-array (only along coordinate d)
                        dA_term = (
                            dAi * Aj * Ak +
                            Ai * dAj * Ak +
                            Ai * Aj * dAk
                        ) * fg[n]

                        # term from derivative of Boys function
                        dF_term = (Ai * Aj * Ak) * (boys_factor * fg[n + 1])

                        ds += dA_term + dF_term

            grad[d] = pref * ds

        return grad

    def _A_term_dcp(self, i, r, u, l1, l2, pax, pbx, cpx, gamma):
        m = i - 2*r - 2*u  # power of cpx in A_term
        if m == 0:
            return 0.0

        # d/dcp (cp^m) = m * cp^(m-1)
        return (-1)**i * binomial_prefactor(i, l1, l2, pax, pbx) * \
            (-1)**u * factorial(i) * (m) * np.power(cpx, m - 1) * \
            np.power(0.25 / gamma, r + u) / factorial(r) / factorial(u) / \
            factorial(m)


    def _A_array_dcd(self, l1, l2, pa, pb, cp, g):
        imax = l1 + l2 + 1
        darr = np.zeros(imax)

        for i in range(imax):
            for r in range(i // 2 + 1):
                for u in range((i - 2*r) // 2 + 1):
                    iI = i - 2*r - u
                    # d/dc = - d/dcp
                    darr[iI] -= self._A_term_dcp(i, r, u, l1, l2, pa, pb, cp, g)

        return darr
