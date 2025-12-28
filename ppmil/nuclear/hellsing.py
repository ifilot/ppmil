import numpy as np
from scipy.special import factorial

from ..util.gto import GTO
from .nuclear_engine import NuclearEngine
from ..math.math import gaussian_product_center
from ..math.gamma_numba import Fgamma

class HellsingNuclearEngine(NuclearEngine):

    def __init__(self, use_kernel=False, lmax=3):
        # always cache factorials
        self._fact = np.array(
            [factorial(i) for i in range(lmax*3)],
            dtype=np.float64
        )

        self._use_kernel = use_kernel
        if self._use_kernel:
            self._compute_kernel(lmax)

    def nuclear_primitive(self, gto1:GTO, gto2:GTO, nucleus):
        """
        Calculate nuclear attraction integral for two GTOs
        """
        return self._nuclear(gto1.p, gto1.o, gto1.alpha,
                             gto2.p, gto2.o, gto2.alpha,
                             nucleus)

    def get_kernel_coefficients(self, l1, l2):
        if self._use_kernel and l1 <= self._lmax and l2 <= self._lmax:
            return self._kernel[l1][l2]
        else:
            return None
        
    def print_kernel_coefficients(self, l1, l2):
        terms = self.get_kernel_coefficients(l1, l2)

        for i,(s,a1,a2,gam,xp,pcxp,coeff) in enumerate(zip(*terms)):
            print('%i: c=%+4.2f  a1=%i  a2=%i  gam=%i  xp=%i  pcxp=%i  mu=%i  u=%i' % ((i+1),s,a1,a2,gam,xp,pcxp,coeff[0],coeff[1]))

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
        
        if self._use_kernel:
            ax, mx = self._A_array_kernel(o1[0], o2[0], alpha1, alpha2, a[0]-b[0], gamma, p[0] - c[0])
            ay, my = self._A_array_kernel(o1[1], o2[1], alpha1, alpha2, a[1]-b[1], gamma, p[1] - c[1])
            az, mz = self._A_array_kernel(o1[2], o2[2], alpha1, alpha2, a[2]-b[2], gamma, p[2] - c[2])
        else:
            ax, mx = self._A_array(o1[0], o2[0], alpha1, alpha2, a[0]-b[0], gamma, p[0] - c[0])
            ay, my = self._A_array(o1[1], o2[1], alpha1, alpha2, a[1]-b[1], gamma, p[1] - c[1])
            az, mz = self._A_array(o1[2], o2[2], alpha1, alpha2, a[2]-b[2], gamma, p[2] - c[2])

        # pre-calculate Fgamma values
        nu_max =  mx[0][0] + my[0][0] + mz[0][0] - (mx[0][1] + my[0][1] + mz[0][1])
        fg = np.array([Fgamma(nu, gamma*rcp2) for nu in range(nu_max+1)])

        s = 0
        for i in range(len(ax)):
            for j in range(len(ay)):
                for k in range(len(az)):
                    nu = mx[i][0] + my[j][0] + mz[k][0] - (mx[i][1] + my[j][1] + mz[k][1])
                    s += ax[i] * ay[j] * az[k] * fg[nu]
        
        return -2.0 * np.pi / gamma * np.exp(-alpha1*alpha2*rab2/gamma) * s

    def _A_array(self, l1, l2, alpha1, alpha2, x, gamma, pcx):
        arr = []
        mu_u = []
        
        pre = np.power(-1, l1+l2) * self._fact[l1] * self._fact[l2]

        for i1 in range(l1//2+1):
            for i2 in range(l2//2+1):
                for o1 in range(l1 - 2*i1+1):
                    for o2 in range(l2 - 2*i2+1):
                        for r in range((o1+o2)//2+1):
                            t1 = np.power(-1, o2+r) * \
                                 self._fact[o1 + o2] / \
                                 np.power(4, i1+i2+r) / \
                                 self._fact[i1] / \
                                 self._fact[i2] / \
                                 self._fact[o1] / \
                                 self._fact[o2] / \
                                 self._fact[r]
                            t2 = np.power(alpha1, o2-i1-r) * \
                                 np.power(alpha2, o1-i2-r) * \
                                 np.power(x, o1+o2-2*r) / \
                                 self._fact[l1 -2*i1 - o1] / \
                                 self._fact[l2 -2*i2 - o2] / \
                                 self._fact[o1 + o2 - 2*r]
                            
                            mu_x = l1 + l2 - 2*(i1+i2) - (o1+o2)
                            for u in range(mu_x//2+1):
                                t3 = np.power(-1, u) * \
                                     self._fact[mu_x] * \
                                     np.power(pcx, mu_x - 2*u) / \
                                     np.power(4, u) / \
                                     self._fact[u] / \
                                     self._fact[mu_x - 2*u] / \
                                     np.power(gamma, o1 + o2 - r + u)

                                arr.append(t1 * t2 * t3)
                                mu_u.append((mu_x, u))

        return pre * np.asarray(arr), mu_u
    
    def _A_array_kernel(self, l1, l2, alpha1, alpha2, x, gamma, pcx):
        terms = self._kernel[l1][l2]
        arr = np.empty(len(terms[0]))

        if l1 == l2 == 0:
            return np.asarray([1.0]), [(0,0)]

        # 1: c=-1.00  a1=0  a2=0  gam=0  xp=0  pcxp=1  mu=1  u=0
        # 2: c=-1.00  a1=0  a2=1  gam=1  xp=1  pcxp=0  mu=0  u=0
        if l1 == 1 and l2 == 0:
            return np.asarray([-pcx, -alpha2/gamma*x]), [(1,0),(0,0)]

        # 1: c=-1.00  a1=0  a2=0  gam=0  xp=0  pcxp=1  mu=1  u=0
        # 2: c=+1.00  a1=1  a2=0  gam=1  xp=1  pcxp=0  mu=0  u=0
        if l1 == 0 and l2 == 1:
            return np.asarray([-pcx, alpha1/gamma*x]), [(1,0),(0,0)]

        for i,(s,a1,a2,gam,xp,pcxp,_) in enumerate(zip(*terms)):
            arr[i] = (s * alpha1**a1 * alpha2**a2 / gamma**gam * x**xp * pcx**pcxp)

        return np.asarray(arr), terms[-1]
    
    def _compute_kernel(self, lmax=3):
        self._lmax = lmax
        self._kernel = [[None for _ in range(lmax+1)] for _ in range(lmax+1)]

        for i in range(lmax+1):
            for j in range(lmax+1):
                self._kernel[i][j] = self._calculate_coefficients(i, j)

    def _calculate_coefficients(self, l1, l2):
        scalar_terms = []
        alpha1_terms = []
        alpha2_terms = []
        gamma_terms = []
        x_terms = []
        pcx_terms = []
        mu_u = []

        pre = np.power(-1, l1+l2) * self._fact[l1] * self._fact[l2]

        for i1 in range(l1//2+1):
            for i2 in range(l2//2+1):
                for o1 in range(l1 - 2*i1+1):
                    for o2 in range(l2 - 2*i2+1):
                        for r in range((o1+o2)//2+1):
                            t1 = np.power(-1, o2+r) * \
                                 self._fact[o1 + o2] / \
                                 np.power(4, i1+i2+r) / \
                                 self._fact[i1] / \
                                 self._fact[i2] / \
                                 self._fact[o1] / \
                                 self._fact[o2] / \
                                 self._fact[r]
                            t2 = self._fact[l1-2*i1 - o1] * \
                                 self._fact[l2-2*i2 - o2] * \
                                 self._fact[o1 + o2 - 2*r]
                            
                            mu_x = l1 + l2 - 2*(i1+i2) - (o1+o2)
                            for u in range(mu_x//2+1):
                                t3 = np.power(-1, u) * \
                                     self._fact[mu_x] / \
                                     np.power(4, u) / \
                                     self._fact[u] / \
                                     self._fact[mu_x- 2*u]
                                
                                scalar_terms.append(pre * t1 / t2 * t3)
                                alpha1_terms.append(o2-i1-r)
                                alpha2_terms.append(o1-i2-r)
                                x_terms.append(o1+o2-2*r)
                                gamma_terms.append(o1 + o2 - r + u)
                                pcx_terms.append(mu_x - 2*u)
                                mu_u.append((mu_x, u))
        
        return (np.asarray(scalar_terms, dtype=np.float64), 
                np.asarray(alpha1_terms, dtype=np.int64), 
                np.asarray(alpha2_terms, dtype=np.int64), 
                np.asarray(gamma_terms, dtype=np.int64), 
                np.asarray(x_terms, dtype=np.int64),
                np.asarray(pcx_terms, dtype=np.int64),
                mu_u)
    
    def _nuclear_grad_c(self, a, o1, alpha1, b, o2, alpha2, c):
        """
        Analytic gradient of primitive nuclear attraction integral w.r.t nuclear center c.
        Returns np.ndarray shape (3,): [dV/dcx, dV/dcy, dV/dcz]

        Matches this engine's _nuclear() exactly (including kernel vs non-kernel A-array).
        """
        gamma = alpha1 + alpha2
        p = gaussian_product_center(alpha1, a, alpha2, b)

        rab2 = np.sum((a - b) ** 2)
        # rcp2 uses (c - p)^2, equivalent to (p - c)^2
        cp = c - p
        rcp2 = np.sum(cp ** 2)

        # Build A-arrays and mu bookkeeping
        if self._use_kernel:
            ax, mx = self._A_array_kernel(o1[0], o2[0], alpha1, alpha2, a[0] - b[0], gamma, p[0] - c[0])
            ay, my = self._A_array_kernel(o1[1], o2[1], alpha1, alpha2, a[1] - b[1], gamma, p[1] - c[1])
            az, mz = self._A_array_kernel(o1[2], o2[2], alpha1, alpha2, a[2] - b[2], gamma, p[2] - c[2])
        else:
            ax, mx = self._A_array(o1[0], o2[0], alpha1, alpha2, a[0] - b[0], gamma, p[0] - c[0])
            ay, my = self._A_array(o1[1], o2[1], alpha1, alpha2, a[1] - b[1], gamma, p[1] - c[1])
            az, mz = self._A_array(o1[2], o2[2], alpha1, alpha2, a[2] - b[2], gamma, p[2] - c[2])

        # nu_max definition as in your code
        nu_max = mx[0][0] + my[0][0] + mz[0][0] - (mx[0][1] + my[0][1] + mz[0][1])

        # Need Boys up to nu_max+1 for derivative term
        fg = np.array([Fgamma(nu, gamma * rcp2) for nu in range(nu_max + 2)], dtype=np.float64)

        pref = -2.0 * np.pi / gamma * np.exp(-alpha1 * alpha2 * rab2 / gamma)

        grad = np.zeros(3, dtype=np.float64)

        # Precompute the nu "parts" per axis (so we don't keep indexing tuples)
        mux = np.array([t[0] for t in mx], dtype=np.int64)
        uux = np.array([t[1] for t in mx], dtype=np.int64)
        muy = np.array([t[0] for t in my], dtype=np.int64)
        uuy = np.array([t[1] for t in my], dtype=np.int64)
        muz = np.array([t[0] for t in mz], dtype=np.int64)
        uuz = np.array([t[1] for t in mz], dtype=np.int64)

        # Helper to compute ds/dc_d
        for d in range(3):
            pcx = p[d] - c[d]          # this is your pcx variable
            boys_factor = 2.0 * gamma * pcx  # from dF/dc_d = 2*gamma*pcx*F_{n+1}

            # dA arrays for the differentiated axis only
            if d == 0:
                dax, _ = self._A_array_dcd(o1[0], o2[0], alpha1, alpha2, a[0] - b[0], gamma, p[0] - c[0])
                day = None
                daz = None
            elif d == 1:
                day, _ = self._A_array_dcd(o1[1], o2[1], alpha1, alpha2, a[1] - b[1], gamma, p[1] - c[1])
                dax = None
                daz = None
            else:
                daz, _ = self._A_array_dcd(o1[2], o2[2], alpha1, alpha2, a[2] - b[2], gamma, p[2] - c[2])
                dax = None
                day = None

            ds = 0.0
            for i in range(len(ax)):
                Ai = ax[i]
                dAi = dax[i] if dax is not None else 0.0
                nu_i_num = mux[i]
                nu_i_den = uux[i]

                for j in range(len(ay)):
                    Aj = ay[j]
                    dAj = day[j] if day is not None else 0.0
                    nu_j_num = muy[j]
                    nu_j_den = uuy[j]

                    Aij = Ai * Aj
                    dAij = dAi * Aj + Ai * dAj
                    nu_ij_num = nu_i_num + nu_j_num
                    nu_ij_den = nu_i_den + nu_j_den

                    for k in range(len(az)):
                        Ak = az[k]
                        dAk = daz[k] if daz is not None else 0.0
                        nu_k_num = muz[k]
                        nu_k_den = uuz[k]

                        nu = (nu_ij_num + nu_k_num) - (nu_ij_den + nu_k_den)

                        # A-derivative contribution (only one axis has nonzero dA)
                        dA = dAij * Ak + Aij * dAk
                        term_A = dA * fg[nu]

                        # Boys derivative contribution
                        term_F = (Aij * Ak) * (boys_factor * fg[nu + 1])

                        ds += term_A + term_F

            grad[d] = pref * ds

        return grad


    def _A_array_dcd(self, l1, l2, alpha1, alpha2, x, gamma, pcx):
        """
        Analytic derivative d/dc of the 1D A-array along the axis corresponding to pcx = p - c.

        Since pcx = p - c, we have:
        d/dc (pcx^m) = -m * pcx^(m-1)

        Returns
        -------
        darr : np.ndarray
            Derivative array, same length/order as _A_array/_A_array_kernel output.
        mu_u : list[tuple]
            Same (mu, u) bookkeeping as the parent A-array.
        """
        if self._use_kernel:
            # Use precomputed term structure if available
            terms = self._kernel[l1][l2]
            mu_u = terms[-1]

            # Special-cases must be handled consistently with _A_array_kernel
            if l1 == l2 == 0:
                return np.asarray([0.0]), [(0, 0)]

            if l1 == 1 and l2 == 0:
                # ax = [-pcx, -alpha2/gamma*x]
                # d/dc: d(-pcx)/dc = +1 ; second term independent of c
                return np.asarray([1.0, 0.0]), [(1, 0), (0, 0)]

            if l1 == 0 and l2 == 1:
                # ax = [-pcx, +alpha1/gamma*x]
                # d/dc: d(-pcx)/dc = +1 ; second term independent of c
                return np.asarray([1.0, 0.0]), [(1, 0), (0, 0)]

            # General kernel terms
            scalar, a1e, a2e, game, xpe, pcpe, _ = terms  # last "_" is mu_u list already pulled
            darr = np.empty_like(scalar, dtype=np.float64)

            # d/dc (pcx^m) = -m * pcx^(m-1)
            for i in range(len(scalar)):
                m = int(pcpe[i])
                if m == 0:
                    darr[i] = 0.0
                else:
                    darr[i] = scalar[i] * (alpha1 ** a1e[i]) * (alpha2 ** a2e[i]) \
                            / (gamma ** game[i]) * (x ** xpe[i]) * (-m) * (pcx ** (m - 1))

            return darr, mu_u

        # Non-kernel path: we can reuse your explicit construction but differentiate t3
        arr = []
        mu_u = []

        pre = np.power(-1, l1 + l2) * self._fact[l1] * self._fact[l2]

        for i1 in range(l1 // 2 + 1):
            for i2 in range(l2 // 2 + 1):
                for o1 in range(l1 - 2 * i1 + 1):
                    for o2 in range(l2 - 2 * i2 + 1):
                        for r in range((o1 + o2) // 2 + 1):
                            t1 = np.power(-1, o2 + r) * \
                                self._fact[o1 + o2] / \
                                np.power(4, i1 + i2 + r) / \
                                self._fact[i1] / \
                                self._fact[i2] / \
                                self._fact[o1] / \
                                self._fact[o2] / \
                                self._fact[r]
                            t2 = np.power(alpha1, o2 - i1 - r) * \
                                np.power(alpha2, o1 - i2 - r) * \
                                np.power(x, o1 + o2 - 2 * r) / \
                                self._fact[l1 - 2 * i1 - o1] / \
                                self._fact[l2 - 2 * i2 - o2] / \
                                self._fact[o1 + o2 - 2 * r]

                            mu_x = l1 + l2 - 2 * (i1 + i2) - (o1 + o2)
                            for u in range(mu_x // 2 + 1):
                                # Original:
                                # t3 = (-1)^u * fact[mu_x] * pcx^(mu_x-2u) / (4^u u! (mu_x-2u)!) / gamma^(...)
                                m = mu_x - 2 * u
                                if m == 0:
                                    dt3 = 0.0
                                else:
                                    dt3 = np.power(-1, u) * \
                                        self._fact[mu_x] * (-m) * np.power(pcx, m - 1) / \
                                        np.power(4, u) / self._fact[u] / self._fact[m] / \
                                        np.power(gamma, o1 + o2 - r + u)

                                arr.append(t1 * t2 * dt3)
                                mu_u.append((mu_x, u))

        return pre * np.asarray(arr, dtype=np.float64), mu_u
