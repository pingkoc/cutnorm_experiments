from OptManiMulitBallGBB import *
import numpy as np
from scipy.io import loadmat
import time

# Wen & Yin's Seed for testing.
# Does not exactly replicate the MATLAB result since randn might be diff
seed = 2010
data_source = "./data/"
test_file_name = 'torusg3-82'
data = loadmat(data_source + test_file_name)
n = data['n'][0,0]
m = data['m']
c = data['C']

# Estimation of rank
p = int(max(min(round(np.sqrt(2*n)/2), 20),1))
# Initial point normalized
x0 = np.random.randn(p,n)
nrmx0 = np.sum(x0 * x0, axis=0)
x0 = np.divide(x0, np.sqrt(nrmx0))
tic = time.time()
x,y,out = opt_mani_mulit_ball_gbb(x0, maxcut_quad, c, record=0, mxitr=600, gtol=1e-5, xtol=1e-5, ftol=1e-8, tau=1e-3)
toc = time.time()
tsolve = toc-tic
objf2 = -out['fval']
print('name {:10s}, n {:d}, p {:d}, f {:6.4e}, cpu {:4.2f}, itr {:d}, #func eval {:d}, feasi {:3.2e}, ||Hx|| {:3.2e}\n'.format(test_file_name, n, p, objf2, tsolve, out['itr'], out['nfe'], out['feasi'], out['nrmG']))

