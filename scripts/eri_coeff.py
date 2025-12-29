# -*- coding: utf-8 -*-

import sympy as sp
import numpy as np
from sympy.core.function import AppliedUndef

# Scalars
a1 = sp.Symbol('a1')
a2 = sp.Symbol('a2')
a3 = sp.Symbol('a3')
a4 = sp.Symbol('a4')

g1 = sp.Symbol('g1')
g2 = sp.Symbol('g2')

eta = sp.Symbol('eta')
T   = sp.Symbol('T')

# Vector components
ABx = sp.Symbol('ABx')
ABy = sp.Symbol('ABy')
ABz = sp.Symbol('ABz')

CDx = sp.Symbol('CDx')
CDy = sp.Symbol('CDy')
CDz = sp.Symbol('CDz')

PQx = sp.Symbol('PQx')
PQy = sp.Symbol('PQy')
PQz = sp.Symbol('PQz')

F = sp.Function('F')

SYMBOLS = [
    a1, a2, a3, a4,
    g1, g2,
    ABx, CDx, PQx,
    ABy, CDy, PQy,
    ABz, CDz, PQz,
    eta
]

LATEX_NAMES = {
    ABx: r'AB_{x}', ABy: r'AB_{y}', ABz: r'AB_{z}',
    CDx: r'CD_{x}', CDy: r'CD_{y}', CDz: r'CD_{z}',
    PQx: r'PQ_{x}', PQy: r'PQ_{y}', PQz: r'PQ_{z}',
    a1: r'\alpha_{1}', a2: r'\alpha_{2}', a3: r'\alpha_{3}', a4: r'\alpha_{4}',
    g1: r'\gamma_{1}', g2: r'\gamma_{2}',
    eta: r'\eta',
    T: r'T',
}

AXES = ['x', 'y', 'z']

from ppmil import Molecule, GTO
from ppmil import IntegralEvaluator, HellsingElectronRepulsionEngine
from ppmil.eri.teindex import teindex

def main():
    # construct integrator object
    integrator = IntegralEvaluator(None, None, HellsingElectronRepulsionEngine())
    
    levels = [
        [1,1,1,0], # x
        [0,0,0,0], # y
        [0,0,0,0]  # z
       # p s s s
    ]

    b = []
    for i in range(3):
        p = levels[i]
        b.append(integrator._eri_engine._calculate_coefficients(*p))
    
    cmb = []
    flt = []
    nu = []
    for i in range(len(b[0][0])):
        n1 = b[0][1][i][-2] - b[0][1][i][-1]
        for j in range(len(b[1][0])):
            n2 = b[1][1][j][-2] - b[1][1][j][-1]
            for k in range(len(b[2][0])):
                n3 = b[2][1][k][-2] - b[2][1][k][-1]
                flt.append(b[0][0][i] * b[1][0][j] * b[2][0][k])
                coeff = b[0][1][i][:6] + b[1][1][j][:6] + b[2][1][k][:6]
                cart = np.hstack([b[0][1][i][6:9],b[1][1][j][6:9],b[2][1][k][6:9]])
                eta_val = b[0][1][i][-3] + b[1][1][j][-3] + b[2][1][k][-3]
                cmb.append(np.hstack([coeff, cart, eta_val]))
                nu.append(n1 + n2 + n3)
    
    expr = build_expression(flt, cmb, nu)
    expr = order_by_F_function(expr)
    print("\nSimplified expression:\n")
    coeffs = coeffs_by_F(expr)

    orblab = orbital_labels(levels)
    str = r'\begin{align}' + '\n'
    str += r'(%s%s|%s%s) &= ' % tuple(orblab)
    for i,(k,coeff) in enumerate(coeffs.items()):
        str += ('&+' if i != 0 else '') + sp.latex(sp.simplify(coeff), symbol_names=LATEX_NAMES) + r'F_{%i}(T) \\' % i  + '\n'
    str += r'\end{align}' + '\n'
    print(str)

def orbital_labels(levels):
    """
    Given levels (3x4), return labels like ['px', 'py', 's', 's']
    """
    labels = []
    for center in range(4):
        found = False
        for axis, axname in enumerate(AXES):
            if levels[axis][center] == 1:
                labels.append(f"p_{axname}")
                found = True
                break
        if not found:
            labels.append("s")
    return labels

def build_expression(floats, ints, nu):
    expr = 0
    for f, intset, n in zip(floats, ints, nu):
        term = sp.Rational(str(f))
        for base, power in zip(SYMBOLS, intset):
            term *= base**int(power)
        term *= g1**int(n)
        term *= F(int(n), T)     # <-- use Boys order
        expr += term
    return expr

def factor_by_F(expr):
    Fs = list(expr.atoms(sp.Function))
    return sp.factor(expr, *Fs)

def coeffs_by_F(expr):
    """
    Return a dict mapping F(n,T) -> coefficient.
    """
    # Find actual applied Boys functions F(n,T)
    Fs = sorted(
        [f for f in expr.atoms(AppliedUndef) if f.func == F],
        key=lambda f: int(f.args[0])
    )

    if not Fs:
        return {}

    parts = sp.collect(expr, Fs, evaluate=False)

    return {
        Fn: sp.simplify(parts.get(Fn, 0))
        for Fn in Fs
    }

def boys_n(fi):
    n = fi.args[0]
    if isinstance(n, (int, np.integer)):
        return int(n)
    if getattr(n, "is_Integer", False):
        return int(n)
    raise TypeError(f"Non-integer Boys index: {n!r} in {fi}")

def order_by_F_function(expr):
    Fs = sorted(
        [f for f in expr.atoms(sp.Function) if f.func == F],
        key=boys_n
    )
    collected = sp.collect(expr, Fs, evaluate=False)
    return sp.Add(*[Fi * sp.factor(collected.get(Fi, 0)) for Fi in Fs])

def print_terms(floats, ints, nu):
    for f,intset,n in zip(floats, ints, nu):
        print('%f * a1**%i * a2**%i * a3**%i * a4**%i * g1**%i * g2**%i * (Ai-Bi)**%i * (Ci-Di)**%i * (Pi-Qi)**%i * eta**%i * F(%i,T)' % (f, *intset, n))

if __name__ == '__main__':
    main()