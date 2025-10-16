import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import sys #link_arm <>
sys.path.insert(0, "..") #link_arm <>

df_one100 = pd.read_csv('1link_100_dense.csv')
df_one170 = pd.read_csv('1link_170_dense.csv')
df_two = pd.read_csv('2link_dense.csv')
# print(df_one100)
# print(df_one170)
# print(df_two['Step']==1)

# print(df_one100['Step']==1)
# xxxx

# ranked = pd.DataFrame()
order_one100 = df_one100[['Step','Value']].sort_values(by='Step')#.reset_index(drop=True)
order_one170 = df_one170[['Step','Value']].sort_values(by='Step')#.reset_index(drop=True)
order_two = df_two[['Step','Value']].sort_values(by='Step')#.reset_index(drop=True)
order_two170 = order_two[:422]
# print(order_one100)
# print(order_one170)
# print(order_two[:422])



def running_avg(scores,beta=0.9):
    # print('scores: \n', scores.shape)
    running_mean = [] #running mean
    running_std = [] #running variance
    run_score = scores[0] #initialize mean
    run_var = 0 #initialize variance
    for sc in scores:
        run_score = (1.0-beta)*sc + beta*run_score
        run_var = (1.0-beta)*(sc - run_score)**2 + beta*run_var
        running_mean.append(run_score)
        running_std.append(run_var)

    # print('running_mean: \n', np.array(running_mean).shape)
    # print('running_std: \n', np.array(running_std).shape)
    return running_mean,running_std


def sorted_scored(scores,window=10):
    print('scores: \n', scores.shape)
    print('scores[-window:]: \n', scores[-window:])
    xxx
    avg_reward = np.mean(scores[-window:])
    print('avg_reward: \n', avg_reward)
    xxx
    std_reward = np.std(scores[-window:])
    self.cum_running_score.append(avg_reward)
    self.std_running_score.append(std_reward)
    return 

run_one100 = np.array(running_avg(order_one100['Value']))
run_one170 = np.array(running_avg(order_one170['Value']))
run_two170 = np.array(running_avg(order_two170['Value']))
print('run_one100: \n', run_one100)
print('run_one170: \n', run_one170)
print('run_two170: \n', run_two170)
# xxx


# sorted_one100 = np.array(sorted_scored(order_one100['Value']))
# print('sorted_one100: \n', sorted_one100)
# print('sorted_one100: \n', sorted_one100.shape)
# xxx


# plt.scatter(order_one100['Step'],order_one100['Value'],color="mediumorchid",alpha=0.1) #std_vs_mean
# plt.scatter(order_one170['Step'],order_one170['Value'],color="green",alpha=0.1) #std_vs_mean
# plt.scatter(order_two170['Step'],order_two170['Value'],color="orange",alpha=0.1) #std_vs_mean


plt.plot(order_one100['Step'],run_one100[0],color="mediumorchid",alpha=0.8, label="1-link [1.0]") #std_vs_mean
plt.plot(order_one170['Step'],run_one170[0],color="green",alpha=0.8,label="1-link [1.65]") #std_vs_mean
plt.plot(order_two170['Step'],run_two170[0],color="orange",alpha=0.8, label="2-link [0.95,0.7]") #std_vs_mean

# plt.scatter(order_one100['Step'],run_one100[0],marker='.',color="mediumorchid",alpha=0.1) #std_vs_mean
# lower_bound = run_one100[0] - run_one100[1]
# upper_bound = run_one100[0] + run_one100[1]
# plt.fill_between(order_one100['Step'],lower_bound,upper_bound,facecolor="mediumorchid", alpha=0.15,label='std')
# plt.ylabel('running_score')
# plt.grid()
# plt.xlabel('steps')

# plt.scatter(order_one170['Step'],run_one170[0],marker='.',color="green",alpha=0.9) #std_vs_mean
# lower_bound = run_one170[0] - run_one170[1]
# upper_bound = run_one170[0] + run_one170[1]
# plt.fill_between(order_one100['Step'],lower_bound,upper_bound,facecolor="green", alpha=0.15,label='std')
# plt.ylabel('running_score')
# plt.grid()
# plt.xlabel('steps')

# plt.scatter(order_one100['Step'],run_one100[0],marker='.',color="mediumorchid",alpha=0.1) #std_vs_mean
# lower_bound = run_one100[0] - run_one100[1]
# upper_bound = run_one100[0] + run_one100[1]
# plt.fill_between(order_one100['Step'],lower_bound,upper_bound,facecolor="mediumorchid", alpha=0.15,label='std')
# plt.ylabel('running_score')
# plt.grid()
# plt.xlabel('steps')

plt.legend()
plt.show()

