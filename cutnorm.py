from OptManiMulitBallGBB import *
import numpy as np
from math import gcd
import time

def cutnorm_round(U, V, C, max_iter):
    '''
    Adopted from David Koslicki
    '''
    (p,n) = U.shape
    approx_opt = 0
    uis_opt = np.zeros(n)
    vjs_opt = np.zeros(n)

    for i in range(max_iter):
        g = np.random.randn(p)
        uis = np.sign(g @ U)
        vjs = np.sign(g @ V)

        # Approx
        approx = abs(np.sum(C * (uis.T @ vjs)))

        if approx > approx_opt:
            approx_opt = approx
            uis_opt = uis
            vjs_opt = vjs

    # Cutnorm is 1/4 of infinity norm
    approx_opt = approx_opt/4.
    return approx_opt, uis_opt, vjs_opt

def cutnorm_sets(uis, vjs):
    S = -1*uis[-1]*uis[:-1]
    T = -1*vjs[-1]*vjs[:-1]

    S = (S + 1)/2
    T = (T + 1)/2
    return S, T


def cutnorm(A, B, w1=None, w2=None):
    # Input checking
    if A.ndim != 2 or B.ndim != 2:
        raise ValueError("A and B must be 2D matrices")
    n, n2 = np.shape(A)
    m, m2 = np.shape(B)
    if n != n2 or m != m2:
        raise ValueError("A and B must be square matrices")
    if (w1 is None and w2 is not None) or (w1 is not None and w2 is None):
        raise ValueError("Weight vectors need to be provided for both matrices")
    if (n != len(w1)) or (m != len(w2)):
        raise ValueError("Weight vectors need to be of same length as their corresponding matrix")

    # TODO: Perhaps weighted and diff size matrices can be merged
    if w1 is not None:
        v1 = np.hstack((0, np.cumsum(w1)[:-1], 1.))
        v2 = np.hstack((0, np.cumsum(w2)[:-1], 1.))
        v = np.unique(np.hstack((v1,v2)))
        w = np.diff(v)
        n1 = len(w)

        a = np.zeros(n1, dtype=np.int32)
        b = np.zeros(n1, dtype=np.int32)
        for i in range(n1-1):
            val = (v[i] + v[i+1])/2
            a[i] = np.argwhere(v1>val)[0] - 1
            b[i] = np.argwhere(v2>val)[0] - 1
        # Last element is always itself in new weights
        a[-1] = len(w1)-1
        b[-1] = len(w2)-1
        A_sq = A[a]
        A_sq = A_sq[:,a]
        B_sq = B[b]
        B_sq = B_sq[:,b]
        C = (A_sq - B_sq)

        # Normalize C according to weights
        C = C*(np.outer(w,w))

        perf, perf2, S, T = compute_square_cutnorm(C)

    else:
        if n == m:
            n1 = n
            w = np.ones(n)/n
            C = (A - B)/(n1*n1) # Normalized

            perf, perf2, S, T = compute_square_cutnorm(C)

        else:
            d = gcd(n,m)
            k = n/d
            l = m/d
            c = k + l - 1
            v1 = np.arange(k)/n
            v2 = np.arange(1, l+1)/m
            v = np.hstack((v1, v2))
            np.sort(v)
            w = np.diff(v)
            w = np.tile(w, d)

            # Create matrix of differences
            n1 = len(w)
            vals = np.tile(v[:-1], d) + np.floor(np.arange(n1)/c)/d + 1.0/(2*n1)
            a = np.floor(vals*n).astype(int)
            b = np.floor(vals*m).astype(int)
            A_sq = A[a]
            A_sq = A_sq[:,a]
            B_sq = B[b]
            B_sq = B_sq[:,b]
            C = (A_sq - B_sq)

            # Normalize C according to weights
            C = C*(np.outer(w,w))

            perf, perf2, S, T = compute_square_cutnorm(C)

    return perf, perf2, S, T, w

def compute_square_cutnorm(C):
    n1 = len(C)
    C_col_sum = np.sum(C, axis=0)
    C_row_sum = np.sum(C, axis=1)
    C_tot = np.sum(C_col_sum)
    # Transformation to preserve cutnorm and enforces infinity one norm = 4*cutnorm
    C = np.c_[C, -1.0*C_row_sum]
    C = np.r_[C, [np.concatenate((-1.0*C_col_sum, [C_tot]))]]

    # Modify rank estimation
    p = int(max(min(round(np.sqrt(2*n1)/2), 100),1))

    # Dim for augmented matrix for SDP
    n2 = 2*n1+2

    # Initiali point normalized
    x0 = np.random.randn(p,n2)
    nrmx0 = np.sum(x0 * x0, axis=0)
    x0 = np.divide(x0, np.sqrt(nrmx0))

    tic = time.time()
    x,g,out = opt_mani_mulit_ball_gbb(x0, cutnorm_quad, C, record=0, mxitr=600, gtol=1e-8, xtol=1e-8, ftol=1e-10, tau=1e-3)
    toc = time.time()
    tsolve = toc-tic

    U = x[:, :n2//2]
    V = x[:, n2//2:]
    objf2 = np.abs(np.sum(C * (U.T @ V)))/4.0

    perf = [n2, p, objf2, tsolve, out['itr'], out['nfe'], out['feasi'], out['nrmG']]

    # Gaussian Rounding
    tic = time.time()
    (objf_lower, uis, vjs) = cutnorm_round(U, V, C, 10000)
    toc = time.time()
    tsolve = toc-tic

    (S, T) = cutnorm_sets(uis, vjs)
    perf2 = [objf_lower, tsolve]

    return perf, perf2, S, T


a = np.ones((3,3))
b = np.ones((6,6))
b[2:-2, 2:-2] = 0
# print((a-b)/(6*6))
# print(cutnorm(a,b,1,1))
perf, perf2, S, T, w = cutnorm(a,b,np.ones(3)/3,np.ones(6)/6)
print(perf2)
