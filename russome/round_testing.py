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

# Compute hamming between uis and vjs
uis_opt = perf2[3]
vjs_opt = perf2[4]
uis_list = perf2[5]
vjs_list = perf2[6]
uis_hamming = np.array([np.sum(np.not_equal(uis_opt, uis_k))
                        for uis_k in uis_list])
vjs_hamming = np.array([np.sum(np.not_equal(vjs_opt, vjs_k))
                        for vjs_k in vjs_list])

# Calculate mode for plot seperation
mode_approx = np.median(perf2[2])
approx_bins = np.linspace(np.min(perf2[2]), np.max(perf2[2]), num=400)
approx_high = np.array([approx for approx in perf2[2] if approx >= mode_approx])
approx_low = np.array([approx for approx in perf2[2] if approx < mode_approx])
approx_bins_high = np.array([bin_val for bin_val in approx_bins
                             if bin_val >= mode_approx])
approx_bins_low = np.array([bin_val for bin_val in approx_bins
                            if bin_val < mode_approx])
uis_hamming_high = np.array([uis_hamm for k,uis_hamm in enumerate(uis_hamming)
                             if perf2[2][k] >= mode_approx])
uis_hamming_low = np.array([uis_hamm for k,uis_hamm in enumerate(uis_hamming)
                             if perf2[2][k] < mode_approx])
vjs_hamming_high = np.array([vjs_hamm for k,vjs_hamm in enumerate(vjs_hamming)
                             if perf2[2][k] >= mode_approx])
vjs_hamming_low = np.array([vjs_hamm for k,vjs_hamm in enumerate(vjs_hamming)
                             if perf2[2][k] < mode_approx])

# Plotting Histogram for Cutnorm Val and UV
f, axarr = plt.subplots(3)
axarr[0].hist(approx_high, label="Rounding Cutnorm Vals Upper 50%",
              bins=approx_bins_high)
axarr[0].hist(approx_low, label="Rounding Cutnorm Vals Lower 50%",
              bins=approx_bins_low)
axarr[0].axvline(x=[perf[2]], label="SDP Solution", c='r')
axarr[0].set_xlabel("Cutnorm Val")
axarr[0].set_ylabel("Frequency")
axarr[0].legend()

axarr[1].hist(uis_hamming_high, label="Hamming distance uis to uis_opt Upper 50%",
              bins=200)
axarr[1].hist(uis_hamming_low, label="Hamming distance uis to uis_opt Lower 50%",
              bins=200)
axarr[1].set_xlabel("Hamming Distance out of " +
                    str(uis_opt.shape[0]))
axarr[1].set_ylabel("Frequency")
axarr[1].legend()

axarr[2].hist(vjs_hamming_high, label="Hamming distance vjs to vjs_opt Upper 50%",
              bins=200)
axarr[2].hist(vjs_hamming_low, label="Hamming distance vjs to vjs_opt Lower 50%",
              bins=200)
axarr[2].set_xlabel("Hamming Distance out of " +
                    str(vjs_opt.shape[0]))
axarr[2].set_ylabel("Frequency")
axarr[2].legend()

plt.suptitle("Cutnorm rounding between " + filenames[i] + " and " + filenames[j] + " with lambda=" + str(lamb), fontsize=15)
plt.tight_layout()
plt.show()
