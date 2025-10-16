import numpy as np
import pandas as pd

import sys #link_arm <>
import os
sys.path.insert(0, "..") #link_arm <>

from mi_estimate_strap import main as statistic

link_type = 2 #1
reward_type = 'sparse'
length = 170 #170

if link_type == 1:
    data = np.load('{}_link_{}_{}_dataset_total.npy'.format(link_type,reward_type,length))
else:
    data = np.load('{}_link_{}_dataset_total.npy'.format(link_type,reward_type))

df = pd.DataFrame(data)#[:10]
# print('df: \n',df.shape)
# xxx

samples = df.transpose()
# print('samples: \n',samples)


# J = statistic(samples.to_numpy(),link_type,reward_type,length)
# print('J: ', J)
# xxxx

def bootstrap_stat(data, statistic, B=1000, alpha=0.05, random_state=None):
    rng = np.random.default_rng(random_state)
    data = np.asarray(data)

    # print('data: \n ', pd.DataFrame(data).transpose())
    # xxx
    n = data.shape[0]
    # print('n: ', n)
    # xxx

    pic_hat,poic_hat = statistic(data,link_type,reward_type,length) #original_estimate
    # print('pic_hat: ', pic_hat)

    pic_star, poic_star = np.empty(B), np.empty(B)
    for b in range(B):
        idx = rng.integers(0,n,size=n) # sample with replacement
        # print('idx: ', idx)
        sample = data[idx]
        # print('sample: \n ', pd.DataFrame(sample).transpose())
        pic_star[b],poic_star[b] = statistic(sample,link_type,reward_type,length)
            
    pic_mean_star = pic_star.mean()
    pic_sd = pic_star.std(ddof=1)
    pic_bias = pic_mean_star - pic_hat

    poic_mean_star = poic_star.mean()
    poic_sd = poic_star.std(ddof=1)
    poic_bias = poic_mean_star - poic_hat

    #percentile CI
    pic_lower = np.percentile(pic_star, 100*(alpha/2))
    pic_upper = np.percentile(pic_star, 100*(1-alpha/2))
    pic_ci = (pic_lower,pic_upper)

    poic_lower = np.percentile(poic_star, 100*(alpha/2))
    poic_upper = np.percentile(poic_star, 100*(1-alpha/2))
    poic_ci = (poic_lower,poic_upper)

    # return {
    #     "point_pic_est": pic_hat,
    #     "point_poic_est": poic_hat,
    #     "pic_std": pic_sd,
    #     "poic_std": poic_sd,
    #     "pic_bias": pic_bias,
    #     "poic_bias": poic_bias,
    #     "pic_ci": pic_ci,
    #     "poic_ci": poic_ci, 
    #     }

    return {'pic': pic_star, 'poic': poic_star}


## running_script

out = bootstrap_stat(df.to_numpy(),statistic)

df = pd.DataFrame.from_dict(out)
if link_type == 1: name = './bootstrapping/permutations_{}link_{}_{}.npy'.format(link_type,length,reward_type)
else: name = './bootstrapping/permutations_{}link_{}.npy'.format(link_type,reward_type)

# if link_type == 1: name = './bootstrapping/{}link_{}_{}.npy'.format(link_type,length,reward_type)
# else: name = './bootstrapping/{}link_{}.npy'.format(link_type,reward_type)

# print('out: \n', out)
# with open(name, 'w') as file:
#     file.write(str(out))

print('df: \n', df)
np.save(name,df)
