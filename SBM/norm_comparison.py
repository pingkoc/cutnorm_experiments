import os
import sys
sys.path.append(os.getcwd() + '/../')
from cutnorm import cutnorm
from sbm import *
from distort import add_gaussian_noise, shift
import numpy as np
import matplotlib.pyplot as plt

# prob_mat_A = [0.9, 0.2, 0.9, 0.4]
prob_mat_A = [0.5, 1.]
prob_mat_B = [0.5]
n_com_A = len(prob_mat_A)
n_com_B = len(prob_mat_B)

# Changing resolution
n_res = 4
logn_list = np.arange(n_res)+2
n_list = np.power(2, logn_list)

cutnorm_list = np.zeros(n_res)
l1_list = np.zeros(n_res)
l2_list = np.zeros(n_res)
linf_list = np.zeros(n_res)

for i, n in enumerate(n_list):
    print("Computing norm for n=", n)
    A = sbm_autoregressive([n//n_com_A]*n_com_A, prob_mat_A)
    A = make_symmetric_triu(A)
    B = sbm_autoregressive([n//n_com_B]*n_com_B, prob_mat_B)
    B = make_symmetric_triu(B)
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

plt.subplot(121)
plt.imshow(A-B, interpolation='gaussian', cmap=plt.get_cmap("coolwarm"))
plt.title("Underlying difference graphon")
plt.axis('off')

plt.subplot(122)
plt.plot(logn_list, cutnorm_list, label="Cutnorm")
plt.plot(logn_list, l1_list, label="L1")
plt.plot(logn_list, l2_list, label="L2")
plt.plot(logn_list, linf_list, label="Linf")
plt.xlabel("log2(n), n dim of matrix")
plt.ylabel("Norm")
plt.legend()
plt.title("Various norms through increasing resolution")
plt.show()

# Shifting
n = 1024
n_rolls = 8
cutnorm_list = np.zeros(n_rolls)
l1_list = np.zeros(n_rolls)
l2_list = np.zeros(n_rolls)
linf_list = np.zeros(n_rolls)
roll_logn_list = np.arange(n_rolls)+2
roll_list = np.power(2, roll_logn_list)

A = sbm_autoregressive([n//n_com_A]*n_com_A, prob_mat_A)
A = make_symmetric_triu(A)
B = sbm_autoregressive([n//n_com_B]*n_com_B, prob_mat_B)
B = make_symmetric_triu(B)

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

plt.subplot(121)
plt.imshow(A-B, interpolation='gaussian', cmap=plt.get_cmap("coolwarm"))
plt.title("Unshifted difference graphon")
plt.axis('off')

plt.subplot(122)
plt.plot(roll_logn_list, cutnorm_list, label="Cutnorm")
plt.plot(roll_logn_list, l1_list, label="L1")
plt.plot(roll_logn_list, l2_list, label="L2")
plt.plot(roll_logn_list, linf_list, label="Linf")
plt.xlabel("log2(Shift amount)")
plt.ylabel("Norm")
plt.legend()
plt.title("Norms under shifting")
plt.show()
