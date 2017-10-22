from OptManiMulitBallGBB import *
import numpy as np
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


def cutnorm(A, B, w1, w2):
    # Input checking
    if A.ndim != 2 or B.ndim != 2:
        raise ValueError("A and B must be 2D matrices")
    n, n2 = np.shape(A)
    m, m2 = np.shape(A)
    if n != n2 or m != m2:
        raise ValueError("A and B must be square matrices")

    if n == m:
        n1 = n
        w = np.ones(n)/n
        C = A - B
        C = C/(n1*n1) # Normalize

        C_col_sum = np.sum(C, axis=0)
        C_row_sum = np.sum(C, axis=1)
        C_tot = np.sum(C_col_sum)
        # Transformation to preserve cutnorm and enforces infinity one norm = 4*cutnorm
        C = np.c_[C, -1.0*C_row_sum]
        C = np.r_[C, [np.concatenate((C_col_sum, [C_tot]))]]

        # Modify rank estimation
        p = int(max(min(round(np.sqrt(2*n1)/2), 100),1))

        # Dim for augmented matrix for SDP
        n2 = 2*n1+2
        # New augmented matrix (Opt: Use sparse matrix representation)
        C2 = np.c_[np.zeros((n2//2, n2//2)), C]
        C2 = np.r_[C2, np.c_[C, np.zeros((n2//2, n2//2))]]

        # Initiali point normalized
        x0 = np.random.randn(p,n2)
        nrmx0 = np.sum(x0 * x0, axis=0)
        x0 = np.divide(x0, np.sqrt(nrmx0))

        tic = time.time()
        x,g,out = opt_mani_mulit_ball_gbb(x0, maxcut_quad, C2, record=0, mxitr=600, gtol=1e-8, xtol=1e-8, ftol=1e-10, tau=1e-3)
        toc = time.time()
        tsolve = toc-tic

        U = x[:p, :(n1+1)]
        V = x[:p, (n1+1):n2]
        objf2 = np.abs(np.sum(C * (U.T @ V)))/4.0

        perf = [n2, p, objf2, tsolve, out['itr'], out['nfe'], out['feasi'], out['nrmG']]

        # Gaussian Rounding
        tic = time.time()
        (objf_lower, uis, vjs) = cutnorm_round(U, V, C, 10000)
        toc = time.time()
        tsolve = toc-tic

        (S, T) = cutnorm_sets(uis, vjs)
        perf2 = [objf_lower, tsolve]
    return perf, perf2, S, T, w

a = np.ones((6,6))
b = np.ones((6,6))
b[2:-2, 2:-2] = 0
print((a-b)/(6*6))
print(cutnorm(a,b,1,1))
