import numpy as np
import pandas as pd



import sys #link_arm <>
sys.path.insert(0, "..") #link_arm <>

# data = np.load('/home/rea/pic/examples/results/dist_normal_layers2_units5/one_link.npy')
data = np.load('one_link.npy')

df = pd.DataFrame(data)

print('numpy: \n',data)
print('pandas: \n',df)
print('size:',data.shape)