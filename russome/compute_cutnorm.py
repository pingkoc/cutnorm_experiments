import os
import sys
import numpy as np
sys.path.append(os.getcwd() + '/../')
from cutnorm import cutnorm

# Get all files and sort
filenames = os.listdir(os.getcwd() + '/data/')
filenames.sort()
total_num_files = len([name for name in os.listdir(os.getcwd() + '/data/')])

# Storage for all networks
nets = np.zeros((total_num_files, 630, 630))

print("Computing correlation coefficient for all data")
for i, filename in enumerate(filenames):
    data = np.loadtxt(os.getcwd() + '/data/' + filename)
    (n, m) = data.shape
    nets[i, :, :] = np.corrcoef(data.T)

dist1 = np.zeros((total_num_files, total_num_files))
dist2 = np.zeros((total_num_files, total_num_files))
for i in range(total_num_files-1):
    for j in range(i+1, total_num_files):
        print("Computing cutnorm between " + filenames[i] + " and " + filenames[j])
        A = nets[i]
        B = nets[j]
        [perf, perf2, S, T, w] = cutnorm(A,B)
        dist1[i,j] = perf[2]
        dist2[i,j] = perf2[0]
np.save("corrdist_up", dist1)
np.save("corrdist_lw", dist2)

# Compute distance between parts above lambda
for lamb in [.1, .2, .3, .4, .5, .6]:
    for i in range(total_num_files-1):
        for j in range(i+1, total_num_files):
            print("Computing cutnorm between " + filenames[i] + " and " + filenames[j] + " with lambda=" + str(lamb))
            A = nets[i]*(nets[i] > lamb)
            B = nets[j]*(nets[j] > lamb)
            [perf, perf2, S, T, w] = cutnorm(A,B)
            dist1[i,j] = perf[2]
            dist2[i,j] = perf2[0]
    np.save('corrdist_' + str(lamb) + '_up', dist1)
    np.save('corrdist_' + str(lamb) + '_lw', dist2)
