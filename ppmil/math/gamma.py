import math
import numpy as np

# Numba-friendly float constants
_EPS = np.finfo(np.float64).eps
_MIN = np.finfo(np.float64).tiny  # smallest positive normal
_FPMIN = _MIN / _EPS


def gamm_inc(a, x):
    gammap = gammp(a, x)
    gln = gammln(a)
    return math.exp(gln) * gammap


def gammp(a, x):
    ASWITCH = 100.0

    # Numba: do NOT raise; return a sentinel or handle upstream.
    if x < 0.0 or a <= 0.0:
        return math.nan

    if x == 0.0:
        return 0.0
    elif a >= ASWITCH:
        return gammpapprox(a, x, 1)
    elif x < a + 1.0:
        return gser(a, x)
    else:
        return 1.0 - gcf(a, x)


def gser(a, x):
    gln = gammln(a)
    ap = a

    d = 1.0 / a
    s = d

    while True:
        ap += 1.0
        d *= x / ap
        s += d
        if math.fabs(d) < math.fabs(s) * _EPS:
            return s * math.exp(-x + a * math.log(x) - gln)


def gammln(xx):
    # tuple = compile-time constant, good for numba
    cof = (
        57.1562356658629235, -59.5979603554754912,
        14.1360979747417471, -0.491913816097620199,
        0.339946499848118887e-4, 0.465236289270485756e-4,
        -0.983744753048795646e-4, 0.158088703224912494e-3,
        -0.210264441724104883e-3, 0.217439618115212643e-3,
        -0.164318106536763890e-3, 0.844182239838527433e-4,
        -0.261908384015814087e-4, 0.368991826595316234e-5
    )

    if xx <= 0.0:
        return math.nan

    x = xx
    y = xx

    tmp = x + 5.24218750000000000
    tmp = (x + 0.5) * math.log(tmp) - tmp
    ser = 0.999999999999997092

    for c in cof:
        y += 1.0
        ser += c / y

    return tmp + math.log(2.5066282746310005 * ser / x)


def gcf(a, x):
    gln = gammln(a)

    b = x + 1.0 - a
    c = 1.0 / _FPMIN
    d = 1.0 / b
    h = d

    i = 1.0
    while True:
        an = -i * (i - a)
        b += 2.0
        d = an * d + b
        if math.fabs(d) < _FPMIN:
            d = _FPMIN

        c = b + an / c
        if math.fabs(c) < _FPMIN:
            c = _FPMIN

        d = 1.0 / d
        dd = d * c
        h *= dd

        if math.fabs(dd - 1.0) <= _EPS:
            break
        i += 1.0

    return math.exp(-x + a * math.log(x) - gln) * h


def gammpapprox(a, x, psig):
    # use tuples instead of lists
    y = (
        0.0021695375159141994, 0.011413521097787704, 0.027972308950302116,
        0.051727015600492421, 0.082502225484340941, 0.12007019910960293,
        0.16415283300752470, 0.21442376986779355, 0.27051082840644336,
        0.33199876341447887, 0.39843234186401943, 0.46931971407375483,
        0.54413605556657973, 0.62232745288031077, 0.70331500465597174,
        0.78649910768313447, 0.87126389619061517, 0.95698180152629142
    )

    w = (
        0.0055657196642445571, 0.012915947284065419, 0.020181515297735382,
        0.027298621498568734, 0.034213810770299537, 0.040875750923643261,
        0.047235083490265582, 0.053244713977759692, 0.058860144245324798,
        0.064039797355015485, 0.068745323835736408, 0.072941885005653087,
        0.076598410645870640, 0.079687828912071670, 0.082187266704339706,
        0.084078218979661945, 0.085346685739338721, 0.085983275670394821
    )

    a1 = a - 1.0
    if a1 <= 0.0:
        return math.nan

    lna1 = math.log(a1)
    sqrta1 = math.sqrt(a1)
    gln = gammln(a)

    if x > a1:
        xu = max(a1 + 11.5 * sqrta1, x + 6.0 * sqrta1)
    else:
        xu = max(0.0, min(a1 - 7.5 * sqrta1, x - 5.0 * sqrta1))

    s = 0.0
    for j in range(len(y)):
        t = x + (xu - x) * y[j]
        s += w[j] * math.exp(-(t - a1) + a1 * (math.log(t) - lna1))

    ans = s * (xu - x) * math.exp(a1 * (lna1 - 1.0) - gln)

    if psig != 0:
        # return P(a,x)
        if ans > 0.0:
            return 1.0 - ans
        else:
            return -ans
    else:
        # return Q(a,x)
        if ans >= 0.0:
            return ans
        else:
            return 1.0 + ans


def Fgamma(a, x):
    # clamp x away from 0
    x = math.fabs(x)
    if x < 1e-7:
        x = 1e-7

    val = gamm_inc(a + 0.5, x)
    # propagate NaN if bad params
    if val != val:
        return math.nan

    return 0.5 * (x ** (-a - 0.5)) * val