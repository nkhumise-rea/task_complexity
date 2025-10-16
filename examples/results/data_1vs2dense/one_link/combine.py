import numpy as np
import pandas as pd

import sys #link_arm <>
sys.path.insert(0, "..") #link_arm <>

# i = 0
# data = np.load('one_link_{}.npy'.format(i))
# datap = np.load('one_link_{}.npy'.format(i+1))
# print('numpy: \n',data)
# print('size:',data.shape)

# df = pd.DataFrame(data)
# print('pandas: \n',df)
# dfp = pd.DataFrame(datap)
# print('pandas: \n',dfp)

# data2 = np.vstack((data,datap))
# # print('numpy: \n',data2)
# print('size:',data2.shape)

# df2 = pd.DataFrame(data2)
# print('pandas: \n',df2)

for i in range(57):  #57
    # print(i)
    # data = np.load('one_link_{}.npy'.format(i))
    # data2 = np.vstack((data,datap))

    if i == 0:
        data = np.load('one_link_{}.npy'.format(i))
        # print(i)
    else: 
        # print(data)
        # print('i: ', i)
        temp = np.load('one_link_{}.npy'.format(i))
        data = np.vstack((data,temp))
    # dfp = pd.DataFrame(data)
    # print('pandas: \n',dfp)

print('size:',data.shape)
np.save('one_link_total',data)


    # test = np.array([[1,2],[3,4],[5,6]])
    # print(np.vstack((test,test)))
    # xx