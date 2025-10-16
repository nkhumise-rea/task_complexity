import numpy as np
import pandas as pd

import sys #link_arm <>
sys.path.insert(0, "..") #link_arm <>

link_type = 1 #2
reward_type = 'dense' 
for i in range(51):  #57
    # print(i)
    # print('{}-link_{}_dataset_50_total'.format(link_type,reward_type))
    # xx
    if i == 0:
        data = np.load('{}_link_{}_dataset_{}.npy'.format(link_type,reward_type,i))
        # print(i)
    else: 
        # print(data)
        # print('i: ', i)
        temp = np.load('{}_link_{}_dataset_{}.npy'.format(link_type,reward_type,i))
        data = np.vstack((data,temp))
    # dfp = pd.DataFrame(data)
    # print('pandas: \n',dfp)

print('size:',data.shape)
np.save('{}_link_{}_dataset__50_total'.format(link_type,reward_type),data)


    # test = np.array([[1,2],[3,4],[5,6]])
    # print(np.vstack((test,test)))
    # xx