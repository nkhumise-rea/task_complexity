import numpy as np
import pandas as pd

import sys #link_arm <>
import os
sys.path.insert(0, "..") #link_arm <>

from pathlib import Path
directory = Path('/home/rea/pic/examples/data_results/1link_170_s')
extension = ".npy"

link_type = 1
reward_type = 'sparse'
length = 170

i = 0
for file_path in directory.glob(f"*{extension}"):
    path_1 = file_path
    # print('path_1: ',path_1)
    if i == 0:
        data = np.load(path_1)
        
        # df = pd.DataFrame(data)
        # print(df)
        # xx
    else: 
        # print(data)
        # print('i: ', i)

        temp = np.load(path_1)
        data = np.vstack((data,temp))
    i += 1

    if i > 75: #ensure_76_agents
        break

print('done')
print('size:',data.shape)
# xxxx
np.save('../{}_link_{}_{}_dataset_total'.format(link_type,reward_type,length),data)


    # test = np.array([[1,2],[3,4],[5,6]])
    # print(np.vstack((test,test)))
    # xx