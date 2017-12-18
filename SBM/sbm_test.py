import numpy as np
from sbm import *

# print(sbm(np.array([4,2]),np.array([[0.5,0], [0, 0.5]])))

print(sbm_autoregressive(np.array([4,2,1]),np.array([.9, .9, .9])))
