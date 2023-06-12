import math
import numpy as np
import sys

def Fgamma(a, x):
    x = max(abs(x), 1e-7)
    #print(gammainc(a+0.5, x))
    return 0.5 * np.power(x, -a-0.5) * gamm_inc(a+0.5, x)

def gamm_inc(a, x):
    gammap = gammp(a,x)
    gln = gammln(a)
    return np.exp(gln) * gammap

def gammp(a,x):
    ASWITCH = 100

    if x < 0.0 or a <= 0.0:
        raise Exception('Bad value in Fgamma!')
        return 0.0;

    if x == 0.0:
        return 0.0;
    elif a >= ASWITCH:
        return gammpapprox(a,x,1)
    elif x < a + 1.0:
        return gser(a,x)
    else:
        return 1.0 - gcf(a,x)

def gser(a,x):
    eps = sys.float_info.epsilon

    gln = gammln(a)
    ap = a;

    d = s = 1.0/a;
    while True:
        ap += 1
        d *= x / ap
        s += d;
        if abs(d) < abs(s)*eps:
            return s * np.exp(-x + a*np.log(x)-gln)

def gammln(xx):
    
    cof = [57.1562356658629235,-59.5979603554754912,
        14.1360979747417471,-0.491913816097620199,.339946499848118887e-4,
        .465236289270485756e-4,-.983744753048795646e-4,.158088703224912494e-3,
        -.210264441724104883e-3,.217439618115212643e-3,-.164318106536763890e-3,
        .844182239838527433e-4,-.261908384015814087e-4,.368991826595316234e-5]
    if xx <= 0:
        raise Exception('Bad argument in gammln')
        return 0.0;

    y = x = xx
    tmp = x + 5.24218750000000000;
    tmp = (x + 0.5) * np.log(tmp) - tmp;
    ser = 0.999999999999997092;
    
    for c in cof:
        y += 1
        ser += c / y
    
    return tmp + np.log(2.5066282746310005*ser/x)

def gcf(a,x):
    eps = sys.float_info.epsilon
    FPMIN = sys.float_info.min / eps;
 
    gln = gammln(a);
    b = x + 1.0 - a;
    c = 1.0 / FPMIN
    d = 1.0 /b;
    h = d;
    i = 1
    while True:
        an = -i * (i-a)
        b += 2.0
        d = an * d + b
        if abs(d) < FPMIN:
            d = FPMIN
            
        c = b + an /c
        if abs(c) < FPMIN:
            c = FPMIN
            
        d = 1.0 / d
        dd = d * c
        h *= dd
        if abs(dd - 1.0) <= eps:
            break
        i += 1

    return np.exp(-x + a * np.log(x) - gln) * h

def gammpapprox(a, x, psig):

    y = [0.0021695375159141994,
                    0.011413521097787704,0.027972308950302116,0.051727015600492421,
                    0.082502225484340941, 0.12007019910960293,0.16415283300752470,
                    0.21442376986779355, 0.27051082840644336, 0.33199876341447887,
                    0.39843234186401943, 0.46931971407375483, 0.54413605556657973,
                    0.62232745288031077, 0.70331500465597174, 0.78649910768313447,
                    0.87126389619061517, 0.95698180152629142]

    w = [0.0055657196642445571,
                    0.012915947284065419,0.020181515297735382,0.027298621498568734,
                    0.034213810770299537,0.040875750923643261,0.047235083490265582,
                    0.053244713977759692,0.058860144245324798,0.064039797355015485,
                    0.068745323835736408,0.072941885005653087,0.076598410645870640,
                    0.079687828912071670,0.082187266704339706,0.084078218979661945,
                    0.085346685739338721,0.085983275670394821]

    a1 = a - 1.0
    lna1 = np.log(a1)
    sqrta1 = np.sqrt(a1)
    gln = gammln(a)

    if x > a1:
        xu = max(a1 + 11.5 * sqrta1, x + 6.0 * sqrta1)
    else:
        xu = max(0., min(a1 - 7.5 * sqrta1, x - 5.0 * sqrta1))
    s = 0

    for j in range(len(y)):
        t = x + (xu-x)*y[j]
        s += w[j] * np.exp(-(t-a1)+a1*(np.log(t)-lna1))

    ans = s * (xu-x)*np.exp(a1*(lna1-1.)-gln)
    
    if psig:
        if ans > 0.0:
            return 1.0 - ans
        else:
            return -ans
    else:
        if ans >= 0.0:
            return ans
        else:
            return 1.0 + ans