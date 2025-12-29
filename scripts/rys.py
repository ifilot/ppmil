from ppmil.erieval.obara_saika import ObaraSaika
from ppmil import Molecule, GTO, IntegralEvaluator, HellsingElectronRepulsionEngine
import os
from ppmil.math.gamma_numba import Fgamma
import numpy as np

def main():
    # build hydrogen molecule
    mol = Molecule('co', os.path.join(os.path.dirname(__file__), 'data', 'co.xyz'))
    cgfs, nuclei = mol.build_basis(os.path.join(os.path.dirname(__file__), 'data', 'sto3g.json'))

    integrator = IntegralEvaluator(None, None, HellsingElectronRepulsionEngine(True))

    res = np.empty((3))
    for i in range(2,5):
        res[i-2] = integrator.repulsion(cgfs[i], cgfs[0], cgfs[5], cgfs[5])
    print(res)
    print(shell_psss_rys(cgfs[2], cgfs[0], cgfs[5], cgfs[5]))

    res = np.empty((3,3))
    for i in range(2,5):
        for j in range(2,5):
            res[i-2,j-2] = integrator.repulsion(cgfs[i], cgfs[j], cgfs[5], cgfs[5])
    print(res)
    print(shell_ppss_rys(cgfs[2], cgfs[2], cgfs[5], cgfs[5]))

def shell_psss_rys(shell_p, shell_s1, shell_s2, shell_s3):
    """
    Contracted (ps|ss) shell block via Rys quadrature.
    Returns array shape (3,1,1,1).
    """
    buf = np.zeros(3)

    for gpA in shell_p.gtos:
        for gpB in shell_s1.gtos:
            for gsC in shell_s2.gtos:
                for gsD in shell_s3.gtos:
                    prim = primitive_psss_rys(gpA, gpB, gsC, gsD)
                    buf += gpA.norm * gpA.c * gpB.norm* gpB.c * gsC.norm * gsC.c * gsD.norm * gsD.c * prim

    return buf.reshape(3)

def shell_ppss_rys(shell_pA, shell_pB, shell_sC, shell_sD):
    """
    Contracted (pp|ss) shell block via Rys quadrature.
    Returns array shape (3,3,1,1).
    """
    buf = np.zeros((3, 3))

    for gpA in shell_pA.gtos:
        for gpB in shell_pB.gtos:
            for gsC in shell_sC.gtos:
                for gsD in shell_sD.gtos:
                    prim = primitive_ppss_rys(gpA, gpB, gsC, gsD)
                    buf += gpA.norm * gpA.c * gpB.norm* gpB.c * gsC.norm * gsC.c * gsD.norm * gsD.c * prim

    return buf.reshape(3, 3)

def primitive_psss_rys(gpA, gsB, gsC, gsD):
    """
    Primitive (ps|ss) using 1-point Rys quadrature.
    Returns a length-3 vector: [px, py, pz].
    """

    # --- centers ---
    A = gpA.p
    B = gsB.p
    C = gsC.p
    D = gsD.p

    # --- exponents ---
    a = float(gpA.alpha)
    b = float(gsB.alpha)
    c = float(gsC.alpha)
    d = float(gsD.alpha)

    # --- composite quantities ---
    p = a + b
    q = c + d

    P = (a*A + b*B) / p
    Q = (c*C + d*D) / q
    W = (p*P + q*Q) / (p + q)

    PA = P - A
    WP = W - P

    # --- distances ---
    RAB2 = np.dot(A - B, A - B)
    RCD2 = np.dot(C - D, C - D)
    RPQ2 = np.dot(P - Q, P - Q)

    # --- Boys argument ---
    T = (p*q/(p+q)) * RPQ2

    # --- Boys moments ---
    F0 = float(Fgamma(0, T))
    F1 = float(Fgamma(1, T))

    # --- Rys quadrature (1 root) ---
    w  = F0
    u2 = F1 / F0

    # --- prefactor ---
    pref = 2.0 * np.pi**2.5 / (p*q*np.sqrt(p+q))
    pref *= np.exp(-a*b/p * RAB2)
    pref *= np.exp(-c*d/q * RCD2)

    # --- final result ---
    return pref * w * (PA + WP * u2)

def primitive_ppss_rys(gpA, gpB, gsC, gsD):
    import numpy as np

    A, B = gpA.p, gpB.p
    C, D = gsC.p, gsD.p

    a, b = float(gpA.alpha), float(gpB.alpha)
    c, d = float(gsC.alpha), float(gsD.alpha)

    p = a + b
    q = c + d

    P = (a*A + b*B) / p
    Q = (c*C + d*D) / q
    W = (p*P + q*Q) / (p + q)

    PA = P - A
    PB = P - B
    WP = W - P

    T = (p*q/(p+q)) * np.dot(P - Q, P - Q)

    # Boys moments (moments!)
    F0 = Fgamma(0, T)
    F1 = Fgamma(1, T)
    F2 = Fgamma(2, T)
    F3 = Fgamma(3, T)

    # --- build 2-point quadrature ---
    u2, w = rys2_from_boys(F0, F1, F2, F3)

    pref = 2*np.pi**2.5/(p*q*np.sqrt(p+q))
    pref *= np.exp(-a*b/p*np.dot(A-B,A-B))
    pref *= np.exp(-c*d/q*np.dot(C-D,C-D))

    # --- accumulate ---
    buf = np.zeros((3, 3))

    for m in range(2):
        Xm = PA + WP * u2[m]
        Ym = PB + WP * u2[m]

        diag = (1.0 / (2.0 * p)) * (1.0 - (q / (p + q)) * u2[m])

        buf += w[m] * (np.outer(Xm, Ym) + np.eye(3) * diag)

    buf *= pref

    return buf

def rys2_from_boys(F0, F1, F2, F3):
    # moments in t=u^2
    m0, m1, m2, m3 = float(F0), float(F1), float(F2), float(F3)

    a0 = m1 / m0
    beta1 = (m2 / m0) - a0*a0
    if beta1 <= 0.0:
        raise ValueError("Non-positive beta1; check Boys values / T range.")

    den = m2 - 2.0*a0*m1 + a0*a0*m0  # = <(t-a0)^2>
    num = m3 - 2.0*a0*m2 + a0*a0*m1  # = <t (t-a0)^2>
    a1 = num / den

    sb = np.sqrt(beta1)

    # eigenvalues of 2x2 symmetric matrix
    tr = a0 + a1
    det_term = (a0 - a1)*(a0 - a1) + 4.0*beta1
    rt = np.sqrt(det_term)

    t1 = 0.5*(tr + rt)
    t2 = 0.5*(tr - rt)

    # weights from eigenvectors: w_i = m0 * (v0_i)^2
    # For eigenvalue t: v1/v0 = (t - a0)/sb
    def weight_for(t):
        r = (t - a0) / sb
        v0_sq = 1.0 / (1.0 + r*r)
        return m0 * v0_sq

    w1 = weight_for(t1)
    w2 = weight_for(t2)

    # nodes are u^2 values
    u2 = np.array([t1, t2], dtype=float)
    w  = np.array([w1, w2], dtype=float)
    return u2, w

if __name__ == '__main__':
    main()