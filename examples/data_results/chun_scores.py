import numpy as np
import pandas as pd

import sys #link_arm <>
import os
sys.path.insert(0, "..") #link_arm <>

link_type = 2 #
reward_type = 'sparse' #'dense'
length = 170 #170

if link_type == 1:
    data = np.load('{}_link_{}_{}_dataset_total.npy'.format(link_type,reward_type,length))
else:
    data = np.load('{}_link_{}_dataset_total.npy'.format(link_type,reward_type))

df = pd.DataFrame(data)
# print('df: \n',df.shape)
# xxx

samples = df.transpose()
print('samples: \n',samples)

# batch_size
N = 2000
dfs = [samples.iloc[:,i:i+N] for i in range(0,samples.shape[1],N)]
print('len(dfs): ', len(dfs))

for idx, dfi in enumerate(dfs):
    # print('dfi: \n', dfi)
    # print('dfi: \n', dfi.head())

    bag = dfi.transpose().to_numpy()
    print('size: ', dfi.shape)
    # print(bag)
    if link_type == 1:
        np.save('./batches/{}link_{}_{}_batch/_2kBatch_{}.npy'.format(link_type,length,reward_type,idx), bag)
    else:
        np.save('./batches/{}link_{}_batch/_2kBatch_{}.npy'.format(link_type,reward_type,idx), bag)
