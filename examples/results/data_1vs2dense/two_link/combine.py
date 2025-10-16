import numpy as np
import pandas as pd

import sys #link_arm <>
sys.path.insert(0, "..") #link_arm <>

for i in range(57):  
    if i == 0:
        data = np.load('two_link_{}.npy'.format(i))
    else: 
        temp = np.load('two_link_{}.npy'.format(i))
        data = np.vstack((data,temp))

print('size:',data.shape)
np.save('two_link_total',data)