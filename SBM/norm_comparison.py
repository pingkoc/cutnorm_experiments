import os
import sys
sys.path.append(os.getcwd() + '/../')
from cutnorm import cutnorm
from sbm import *
from distort import add_gaussian_noise, shift
import numpy as np
import matplotlib.pyplot as plt

# prob_mat = [0.9, 0.2, 0.9, 0.4]
prob_mat = [0.5]
n_com = len(prob_mat)

# Changing resolution
n_points = 12
logn_list = np.arange(n_points)+2
n_list = np.power(2, logn_list)

cutnorm_list = np.zeros(n_points)
l1_list = np.zeros(n_points)
l2_list = np.zeros(n_points)
linf_list = np.zeros(n_points)

for i, n in enumerate(n_list):
    print("Computing norm for n=", n)
    A = sbm_autoregressive([n//n_com]*n_com, prob_mat)
    A = make_symmetric_triu(A)
    B = sbm_autoregressive_prob([n//n_com]*n_com, prob_mat)
    [perf, perf2, S, T, w] = cutnorm(A, B, max_round_iter=100)
    cutnorm_list[i] = perf2[0]

    # L1 Norm
    C = ((A-B)/n**2).flatten()
    l1_list[i] = np.linalg.norm(C, ord=1)
    l2_list[i] = np.linalg.norm(C, ord=2)
    linf_list[i] = np.linalg.norm(C, ord=np.inf)

print(l1_list)
print(l2_list)
print(linf_list)
print(cutnorm_list)
plt.plot(logn_list, cutnorm_list, label="Cutnorm")
plt.plot(logn_list, l1_list, label="L1")
plt.plot(logn_list, l2_list, label="L2")
plt.plot(logn_list, linf_list, label="Linf")
plt.xlabel("log2(n), n dim of matrix")
plt.ylabel("Norm")
plt.legend()
plt.title("Norms for SBMs of different resolution")
plt.show()

# Shifting
n = 1024
n_points = 8
cutnorm_list = np.zeros(n_points)
l1_list = np.zeros(n_points)
l2_list = np.zeros(n_points)
linf_list = np.zeros(n_points)
roll_logn_list = np.arange(n_points)+2
roll_list = np.power(2, roll_logn_list)

A = sbm_autoregressive([n//n_com]*n_com, prob_mat)
A = make_symmetric_triu(A)
B = sbm_autoregressive_prob([n//n_com]*n_com, prob_mat)

for i, roll_val in enumerate(roll_list):
    print("Computing norm for roll_val=", roll_val)
    rolled_A = shift(A, roll_val)
    [perf, perf2, S, T, w] = cutnorm(rolled_A, B, max_round_iter=100)
    cutnorm_list[i] = perf2[0]

    # L1 Norm
    C = ((rolled_A - B)/n**2).flatten()
    l1_list[i] = np.linalg.norm(C, ord=1)
    l2_list[i] = np.linalg.norm(C, ord=2)
    linf_list[i] = np.linalg.norm(C, ord=np.inf)

print(l1_list)
print(l2_list)
print(linf_list)
print(cutnorm_list)
plt.plot(roll_logn_list, cutnorm_list, label="Cutnorm")
plt.plot(roll_logn_list, l1_list, label="L1")
plt.plot(roll_logn_list, l2_list, label="L2")
plt.plot(roll_logn_list, linf_list, label="Linf")
plt.xlabel("log2(Shift amount)")
plt.ylabel("Norm")
plt.legend()
plt.title("Norms for SBMs under shifting")
plt.show()
