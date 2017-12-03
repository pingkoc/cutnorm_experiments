import os
import sys
import numpy as np
sys.path.append(os.getcwd() + '/../')
from cutnorm import cutnorm
import matplotlib.pyplot as plt

# Get all files and sort
filenames = os.listdir(os.getcwd() + '/data/')
filenames.sort()
total_num_files = len([name for name in os.listdir(os.getcwd() + '/data/')])
print("Total num files: " + str(total_num_files))

# Storage for all networks
nets = np.zeros((total_num_files, 630, 630))

# Randomly select
i = np.random.randint(total_num_files-1)
j = np.random.randint(low=i+1, high=total_num_files)
print("Selected " + str(filenames[i]) + " and " + str(filenames[j]))

print("Computing correlation coefficient for two randomly selected data")
for k in [i,j]:
    data = np.loadtxt(os.getcwd() + '/data/' + filenames[k])
    (n, m) = data.shape
    nets[k, :, :] = np.corrcoef(data.T)

# Getting a pair of random matrix to evaluate the rounding
lamb = 0.1
print("Computing cutnorm between " + filenames[i] + " and " + filenames[j] + " with lambda=" + str(lamb))
A = nets[i]*(nets[i] > lamb)
B = nets[j]*(nets[j] > lamb)
[perf, perf2, S, T, w] = cutnorm(A,B)
print("SDP Solution (Pre Rounding): " + str(perf[2]) + " Rounding Solution: " + str(perf2[0]))
plt.hist(perf2[2], label="Rounding Cutnorm Vals")
plt.axvline(x=[perf[2]], label="SDP Solution", c='r')
plt.xlabel("Cutnorm Val")
plt.ylabel("Frequency")
plt.legend()
plt.title("Cutnorm between " + filenames[i] + " and " + filenames[j] + " with lambda=" + str(lamb))
plt.show()
