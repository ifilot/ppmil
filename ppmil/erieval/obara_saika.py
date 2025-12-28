import numpy as np
import math
from ..math.gamma_numba import Fgamma

class ObaraSaika:
    """
    Minimal Obara-Saika ERI evaluator for (ps|ss)
    """

    # -------------------------
    # Auxiliary integral R_n
    # -------------------------
    @staticmethod
    def auxiliary_R(ga, gb, gc, gd, n):
        A, B = ga.p, gb.p
        C, D = gc.p, gd.p

        a, b = ga.alpha, gb.alpha
        c, d = gc.alpha, gd.alpha

        p = a + b
        q = c + d

        P = (a*A + b*B) / p
        Q = (c*C + d*D) / q

        RPQ2 = np.dot(P - Q, P - Q)
        RAB2 = np.dot(A - B, A - B)
        RCD2 = np.dot(C - D, C - D)

        T = (p*q / (p + q)) * RPQ2

        prefactor = (
            2.0 * np.pi**2.5 /
            (p * q * np.sqrt(p + q))
        )

        KAB = np.exp(-a*b/p * RAB2)
        KCD = np.exp(-c*d/q * RCD2)

        return prefactor * KAB * KCD * Fgamma(n, T)

    # -------------------------
    # Primitive (ss|ss)
    # -------------------------
    @staticmethod
    def primitive_ssss(ga, gb, gc, gd):
        return ObaraSaika.auxiliary_R(ga, gb, gc, gd, n=0)

    # -------------------------
    # Primitive (ps|ss)
    # -------------------------
    @staticmethod
    def primitive_psss(gp, gs1, gs2, gs3):
        """
        Returns np.array([px, py, pz]) for primitive quartet
        """

        A, B = gp.p, gs1.p
        C, D = gs2.p, gs3.p

        a, b = gp.alpha, gs1.alpha
        c, d = gs2.alpha, gs3.alpha

        p = a + b
        q = c + d

        P = (a*A + b*B) / p
        Q = (c*C + d*D) / q
        W = (p*P + q*Q) / (p + q)

        PA = P - A
        WP = W - P

        R0 = ObaraSaika.auxiliary_R(gp, gs1, gs2, gs3, n=0)
        R1 = ObaraSaika.auxiliary_R(gp, gs1, gs2, gs3, n=1)

        buf = np.zeros(3)
        for i in range(3):
            buf[i] = PA[i] * R0 + WP[i] * R1

        return buf

    # -------------------------
    # Contracted shell (ps|ss)
    # -------------------------
    @staticmethod
    def shell_psss(shell_p, shell_s1, shell_s2, shell_s3):
        """
        Returns block with shape (3,1,1,1)
        """

        buf = np.zeros((3,1,1,1))

        for gp in shell_p.gtos:
            for gs1 in shell_s1.gtos:
                for gs2 in shell_s2.gtos:
                    for gs3 in shell_s3.gtos:

                        prim = ObaraSaika.primitive_psss(
                            gp, gs1, gs2, gs3
                        )

                        coeff = (
                            gp.norm * gp.c * gs1.norm * gs1.c *
                            gs2.norm * gs2.c * gs3.norm * gs3.c
                        )

                        buf[:,0,0,0] += coeff * prim

        return buf