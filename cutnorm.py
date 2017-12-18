from math import gcd
import time
import numpy as np
# from numba import jit
from OptManiMulitBallGBB import opt_mani_mulit_ball_gbb, cutnorm_quad


# @jit(nopython=True, parallel=True)
def cutnorm_round(U: np.ndarray, V: np.ndarray, C: np.ndarray,
                  max_round_iter: int) -> (np.float_, np.ndarray, np.ndarray):
    '''
    Gaussian Rounding for Cutnorm

    The algorithm picks a random standard multivariate gaussian vector
    w in R^p and computes the rounded solution based on sgn(w \dot ui).

    Adopted from David Koslicki's cutnorm rounding code
    https://github.com/dkoslicki/CutNorm
    and Peter Diao's modifications

    Args:
        U: ndarray, (p, n) shaped matrices of relaxed solutions
        V: ndarray, (p, n) shaped matrices of relaxed solutions
        C: ndarray, original (n, n) shaped matrix to compute cutnorm
        max_round_iter: maximum number of rounding operations
    Returns:
        (approx_opt, uis_opt, vjs_opt)
        approx_opt: approximated objective function value
        uis_opt: rounded u vector
        vis_opt: rounded v vector
    '''
    (p, n) = U.shape
    approx_opt = 0
    uis_opt = np.zeros(n)
    vjs_opt = np.zeros(n)
    G = np.random.randn(max_round_iter, p)

    for i in range(max_round_iter):
        g = G[i]
        uis = np.sign(g @ U)
        vjs = np.sign(g @ V)

        # Approx
        approx = np.abs(np.sum(C * np.outer(uis, vjs)))
        # approx = abs(np.sum((C * uis).T * vjs))

        if approx > approx_opt:
            approx_opt = approx
            uis_opt = uis
            vjs_opt = vjs

    # Cutnorm is 1/4 of infinity norm
    approx_opt = approx_opt/4.
    return approx_opt, uis_opt, vjs_opt

def cutnorm_round_testing(U: np.ndarray, V: np.ndarray, C: np.ndarray,
                  max_round_iter: int) -> (np.float_, np.ndarray, np.ndarray):
    '''
    Gaussian Rounding for Cutnorm

    The algorithm picks a random standard multivariate gaussian vector
    w in R^p and computes the rounded solution based on sgn(w \dot ui).

    Adopted from David Koslicki's cutnorm rounding code
    https://github.com/dkoslicki/CutNorm
    and Peter Diao's modifications

    Args:
        U: ndarray, (p, n) shaped matrices of relaxed solutions
        V: ndarray, (p, n) shaped matrices of relaxed solutions
        C: ndarray, original (n, n) shaped matrix to compute cutnorm
        max_round_iter: maximum number of rounding operations
    Returns:
        (approx_opt, uis_opt, vjs_opt)
        approx_opt: approximated objective function value
        uis_opt: rounded u vector
        vis_opt: rounded v vector
    '''
    (p, n) = U.shape
    approx_opt = 0
    uis_opt = np.zeros(n)
    vjs_opt = np.zeros(n)
    G = np.random.randn(max_round_iter, p)

    # Storing results for analysis
    approx_list = np.zeros(max_round_iter)
    uis_list = np.zeros((max_round_iter, n))
    vjs_list = np.zeros((max_round_iter, n))

    for i in range(max_round_iter):
        g = G[i]
        uis = np.sign(g @ U)
        vjs = np.sign(g @ V)

        # Approx
        approx = np.abs(np.sum(C * np.outer(uis, vjs)))
        # approx = abs(np.sum((C * uis).T * vjs))

        # Storing results for analysis
        approx_list[i] = approx/4
        uis_list[i] = uis
        vjs_list[i] = vjs

        if approx > approx_opt:
            approx_opt = approx
            uis_opt = uis
            vjs_opt = vjs

    # Cutnorm is 1/4 of infinity norm
    approx_opt = approx_opt/4.
    return approx_opt, uis_opt, vjs_opt, approx_list, uis_list, vjs_list


def cutnorm_sets(uis: np.ndarray, vjs: np.ndarray) -> (np.ndarray, np.ndarray):
    """
    Generates the curnorm sets from the rounded SDP solutions

    Args:
        uis: ndarray, (n+1, ) shaped array of rounded +- 1 solution
        vis: ndarray, (n+1, ) shaped array of rounded +- 1 solution
    Returns:
        Reconstructed S and T sets that are {1, 0}^n
        (S, T)
        S: Cutnorm set axis = 0
        T: Cutnorm set axis = 1
    """
    S = -1*uis[-1]*uis[:-1]
    T = -1*vjs[-1]*vjs[:-1]

    S = (S + 1)/2
    T = (T + 1)/2
    return S, T


def cutnorm(A, B, w1=None, w2=None, max_round_iter=1000):
    """
    Computes the cutnorm of the differences between the two matrices

    Args:
        A: ndarray, (n, n) matrix
        B: ndarray, (m, m) matrix
        w1: ndarray, (n, 1) array of weights for A
        w2: ndarray, (m, 1) array of weights for B
        max_round_iter: int, number of iterations for gaussian rounding
    Returns:
        (perf, perf2, S, T, w)
        perf: results from OptManiMulitBallGBB
            [n, p, objf, tsolve, itr, nfe, feasi, nrmG]
            n: dimension of matrix
            p: rank
            objf: objective function value
            tsolve: computation time
            itr, nfe, feasi, nrmG: information from OptManiMulitBallGBB
        perf2: results from gaussian rounding
            [objf_lower, tsolve]
            objf_lower: objective function value from gaussian rounding
            tsolve: computation time
        S: Cutnorm set axis = 0
        T: Cutnorm set axis = 1
        w: weight vector
    Raises:
        ValueError: if A and B are of wrong dimension, or if weight vectors
            does not match the corresponding A and B matrices
    """
    # Input checking
    if A.ndim != 2 or B.ndim != 2:
        raise ValueError("A and B must be 2D matrices")
    n, n2 = np.shape(A)
    m, m2 = np.shape(B)
    if n != n2 or m != m2:
        raise ValueError("A and B must be square matrices")
    if (w1 is None and w2 is not None) or (w1 is not None and w2 is None):
        raise ValueError("Weight vectors required for both matrices")
    if (w1 is not None and w2 is not None and
            (n != len(w1) or m != len(w2))):
        raise ValueError("Weight vectors need to have the same lenght "
                         "as the first dimension of the corresponding "
                         "matrices")

    # TODO: Perhaps weighted and diff size matrices can be merged
    if w1 is not None:
        v1 = np.hstack((0, np.cumsum(w1)[:-1], 1.))
        v2 = np.hstack((0, np.cumsum(w2)[:-1], 1.))
        v = np.unique(np.hstack((v1, v2)))
        w = np.diff(v)
        n1 = len(w)

        a = np.zeros(n1, dtype=np.int32)
        b = np.zeros(n1, dtype=np.int32)
        for i in range(n1-1):
            val = (v[i] + v[i+1])/2
            a[i] = np.argwhere(v1 > val)[0] - 1
            b[i] = np.argwhere(v2 > val)[0] - 1
        # Last element is always itself in new weights
        a[-1] = len(w1)-1
        b[-1] = len(w2)-1
        A_sq = A[a]
        A_sq = A_sq[:, a]
        B_sq = B[b]
        B_sq = B_sq[:, b]
        C = (A_sq - B_sq)

        # Normalize C according to weights
        C = C*(np.outer(w, w))

        perf, perf2, S, T = _compute_square_cutnorm(C, max_round_iter)

    else:
        if n == m:
            n1 = n
            w = np.ones(n)/n
            C = (A - B)/(n1*n1)  # Normalized C

            perf, perf2, S, T = _compute_square_cutnorm(C, max_round_iter)

        else:
            d = gcd(n, m)
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
            vals = np.tile(v[:-1], d) + np.floor(np.arange(n1)/c)/d + 1./(2*n1)
            a = np.floor(vals*n).astype(int)
            b = np.floor(vals*m).astype(int)
            A_sq = A[a]
            A_sq = A_sq[:, a]
            B_sq = B[b]
            B_sq = B_sq[:, b]
            C = (A_sq - B_sq)

            # Normalize C according to weights
            C = C*(np.outer(w, w))

            perf, perf2, S, T = _compute_square_cutnorm(C, max_round_iter)

    return perf, perf2, S, T, w


def _compute_square_cutnorm(C: np.ndarray, max_round_iter: int):
    """
    Computes the cutnorm of square matrix C

    Args:
        C: ndarray, (n, n) matrix
    Returns:
        (perf, perf2, S, T)
        perf: results from OptManiMulitBallGBB
            [n, p, objf, tsolve, itr, nfe, feasi, nrmG]
            n: dimension of matrix
            p: rank
            objf: objective function value
            tsolve: computation time
            itr, nfe, feasi, nrmG: information from OptManiMulitBallGBB
        perf2: results from gaussian rounding
            [objf_lower, tsolve]
            objf_lower: objective function value from gaussian rounding
            tsolve: computation time
        S: Cutnorm set axis = 0
        T: Cutnorm set axis = 1
    """
    n1 = len(C)
    C_col_sum = np.sum(C, axis=0)
    C_row_sum = np.sum(C, axis=1)
    C_tot = np.sum(C_col_sum)
    # Transformation to preserve cutnorm and
    # enforces infinity one norm = 4*cutnorm
    C = np.c_[C, -1.0*C_row_sum]
    C = np.r_[C, [np.concatenate((-1.0*C_col_sum, [C_tot]))]]

    # Modify rank estimation
    p = int(max(min(round(np.sqrt(2*n1)/2), 100), 1))

    # Dim for augmented matrix for SDP
    n2 = 2*n1+2

    # Initiali point normalized
    x0 = np.random.randn(p, n2)
    nrmx0 = np.sum(x0 * x0, axis=0)
    x0 = np.divide(x0, np.sqrt(nrmx0))

    tic = time.time()
    x, g, out = opt_mani_mulit_ball_gbb(x0, cutnorm_quad, C, record=0,
                                        mxitr=600, gtol=1e-8, xtol=1e-8,
                                        ftol=1e-10, tau=1e-3)
    toc = time.time()
    tsolve = toc-tic

    U = x[:, :n2//2]
    V = x[:, n2//2:]
    objf2 = np.abs(np.sum(C * (U.T @ V)))/4.0

    perf = [n2, p, objf2, tsolve, out['itr'], out['nfe'],
            out['feasi'], out['nrmG']]

    # Gaussian Rounding
    tic = time.time()

    # (objf_lower, uis, vjs) = cutnorm_round(U, V, C, max_round_iter)
    (objf_lower, uis, vjs, approx_list,
     uis_list, vjs_list) = cutnorm_round_testing(U, V, C,
                                         max_round_iter)
    toc = time.time()
    tsolve = toc-tic

    (S, T) = cutnorm_sets(uis, vjs)
    perf2 = [objf_lower, tsolve, approx_list, uis, vjs, uis_list, vjs_list]

    return perf, perf2, S, T
