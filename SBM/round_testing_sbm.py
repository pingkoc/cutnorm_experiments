import os
import sys
sys.path.append(os.getcwd() + '/../')
from cutnorm import cutnorm
from sbm import sbm
import matplotlib.pyplot as plt

A_prob_mat = [[0.5,0.1], [0.1, 0.5]]
A_community_sizes = [100, 100]
B_prob_mat = [[0.7,0.2], [0.2, 0.3]]
B_community_sizes = [100, 100]

# Print information
print("Making SBM A with Probability Matrix:")
print(A_prob_mat)
print("and community sizes:")
print(A_community_sizes)
print("\nMaking SBM B with Probability Matrix:")
print(B_prob_mat)
print("and community sizes:")
print(B_community_sizes)

A = sbm(A_community_sizes, A_prob_mat)
B = sbm(B_community_sizes, B_prob_mat)

[perf, perf2, S, T, w] = cutnorm(A,B)
print("\nSDP Solution (Pre Rounding): " + str(perf[2]) + " Rounding Solution: " + str(perf2[0]))
plt.hist(perf2[2], label="Rounding Cutnorm Vals")
plt.axvline(x=[perf[2]], label="SDP Solution", c='r')
plt.xlabel("Cutnorm Val")
plt.ylabel("Frequency")
plt.legend()
plt.title("Cutnorm between two almost block diagonal SBM Matrices of 100x100")
plt.show()
