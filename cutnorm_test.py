import numpy as np
from cutnorm import cutnorm


n = 1000
a = np.ones((n, n))
a_small = np.ones((n//2, n//2))
b = np.ones((n, n))
b[n//3:-n//3, n//3:-n//3] = 0

print("Opt obj function val: " + str(np.sum(a - b)/np.sum(a)))

# Same-dim, not weighted
perf, perf2, S, T, w = cutnorm(a, b)
print("Same Dim, not weighted")
print(perf, perf2)

# Diff-dim, not weighted
perf, perf2, S, T, w = cutnorm(a_small, b)
print("Diff Dim, not weighted")
print(perf, perf2)

# Diff-dim Weighted
perf, perf2, S, T, w = cutnorm(a_small, b, np.ones(n//2)/(n//2), np.ones(n)/n)
print("Diff Dim, weighted")
print(perf, perf2)
