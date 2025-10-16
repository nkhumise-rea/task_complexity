import numpy as np
import pandas as pd

data = np.load('permutations_1link_170_dense.npy')
df = pd.DataFrame(data)
print('df: \n', df)