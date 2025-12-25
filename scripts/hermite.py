from scipy.special import hermite, factorial
import numpy as np
import matplotlib.pyplot as plt

def hermite_exp(x,l):
    res = np.zeros_like(x)
    pre = 2**(-l)
    for i in range(l//2+1):
        H = hermite(l - 2*i)
        f = factorial(l) / factorial(i) / factorial (l - 2*i)
        print(f)
        res += f * H(x)

    return pre * res

l = 6
x = np.linspace(-1, 1, 50)
plt.plot(x, x**l)
plt.plot(x, hermite_exp(x, l), 'o', alpha=0.5)
plt.show()