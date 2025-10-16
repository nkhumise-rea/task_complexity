import numpy as np
import pandas as pd


import sys #link_arm <>
sys.path.insert(0, "..") #link_arm <>

data = np.load('two_link0.npy')

df = pd.DataFrame(data)
print('pandas: \n',df)