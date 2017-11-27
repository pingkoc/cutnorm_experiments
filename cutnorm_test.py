import numpy as np
from cutnorm import cutnorm
import matplotlib.pyplot as plt


n = 1000
a = np.ones((n, n))
a_small = np.ones((n//2, n//2))
b = np.ones((n, n))
b[n//3:-n//3, n//3:-n//3] = 0

print("Opt obj function val: " + str(np.sum(a - b)/np.sum(a)))

# # Same-dim, not weighted
# perf, perf2, S, T, w = cutnorm(a, b)
# print("Same Dim, not weighted")
# print(perf, perf2)

# # Diff-dim, not weighted
# perf, perf2, S, T, w = cutnorm(a_small, b)
# print("Diff Dim, not weighted")
# print(perf, perf2)

# # Diff-dim Weighted
# perf, perf2, S, T, w = cutnorm(a_small, b, np.ones(n//2)/(n//2), np.ones(n)/n)
# print("Diff Dim, weighted")
# print(perf, perf2)

# Different n, plotting
n_list = np.arange(100, 1000, 100)
sdp_t = []
round_t = []
for n_val in n_list:
    print("Calculating for n " + str(n_val))
    a = np.ones((n_val, n_val))
    b = np.ones((n_val, n_val))
    b[n_val//3:-n_val//3, n_val//3:-n_val//3] = 0
    # Same-dim, not weighted
    perf, perf2, S, T, w = cutnorm(a, b)
    sdp_t.append(perf[3])
    round_t.append(perf2[1])

plt.plot(n_list, sdp_t, label="sdp time")
plt.plot(n_list, round_t, label="round time")
plt.xlabel("matrix size")
plt.ylabel("Time (s)")
plt.legend()
plt.show()

