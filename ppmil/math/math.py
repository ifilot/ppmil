import numpy as np
import scipy.special

def gaussian_product_center(alpha1, a, alpha2, b):
    """
    Calculate the position of the product of two Gaussians
    """
    return (alpha1 * a + alpha2 * b) / (alpha1 + alpha2)
    
def binomial_prefactor(s, ia, ib, xpa, xpb):
    sm = 0
    
    for t in range(0, s+1):
        if (s - ia) <= t and t <= ib:
            sm += scipy.special.binom(ia, s-t) * scipy.special.binom(ib, t) * \
                    np.power(xpa, ia-s+t) * np.power(xpb, ib-t)
    
    return sm