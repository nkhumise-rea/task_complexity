import numpy as np
import pandas as pd



import sys #link_arm <>
sys.path.insert(0, "..") #link_arm <>

link_type = 1 #2
reward_type = 'dense'  #'sparse'

# data = np.load('/home/rea/pic/examples/results/dist_normal_layers2_units5/one_link.npy')
data = np.load('{}_link_{}_dataset.npy'.format(link_type,reward_type))

df = pd.DataFrame(data)

print('numpy: \n',data)
print('pandas: \n',df)
print('size:',data.shape)