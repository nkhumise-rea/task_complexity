import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats as stats 
from scipy.stats import pearsonr



d = 1 #pic_dense
# d = 2 #pic_sparse
# d = 0 #pic_all

if d == 1:
    x = np.array([4.0,4.1,4.2])
    y = np.array([20,40,10])
elif d == 2:
    x = np.array([0.085,0.071,0.046])
    y = np.array([9,6,1])

elif d == 5:
    x = np.array([4.0,4.1,4.2])
    y = np.array([20,40,10])

else: 
    x = np.array([0.085,0.071,0.046,4.0,4.1,4.2])
    y = np.array([9,6,1,20,40,10])




plt.plot(x,y,'ro')
plt.ylabel("variance(rewards)")
plt.xlabel('PIC')

corr,pval = pearsonr(x,y) 
plt.title(f"R = {corr:.3f}, p = {pval:.3f}") #,fontsize=f_size)

plt.show()