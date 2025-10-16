import numpy as np
import pandas as pd

import sys #link_arm <>
import os
sys.path.insert(0, "..") #link_arm <>

link_type = 1 #2
reward_type = 'sparse' #'dense'
length = 100 #100
n_batch = 0

data = np.load('{}_link_{}_{}_dataset_total.npy'.format(link_type,reward_type,length))
data2 = np.load('./batches/{}link_{}_{}_batch/_2kBatch_{}.npy'.format(link_type,length,reward_type,n_batch), allow_pickle=True)


df = pd.DataFrame(data)
df2 = pd.DataFrame(data2)

print('df: \n', df)
print('df2: \n', df2)