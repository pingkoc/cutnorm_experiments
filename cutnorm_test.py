import numpy as np
from cutnorm import cutnorm


n = 1000
a = np.ones((n//2, n//2))
b = np.ones((n, n))
b[n//3:-n//3, n//3:-n//3] = 0
perf, perf2, S, T, w = cutnorm(a, b, np.ones(n//2)/(n//2), np.ones(n)/n)
print(perf, perf2, S, T, w)
