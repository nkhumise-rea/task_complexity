import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import sys #link_arm <>
sys.path.insert(0, "..") #link_arm <>

#modules
def running_avg(scores,beta=0.99):
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

def roll_mean(array,window=1):
    return np.convolve(array,np.ones(window)/window, mode="same")

def plotting(grid,mean,std):
        ## Plots
        #fig = plt.figure(figsize=[17,15])
        plt.figure(figsize=[17,15])
        plt.plot(grid, mean,color="tomato",label='agent'  )
        plt.fill_between(grid,
                        mean - std,
                        mean + std,
                        facecolor="tomato", 
                        alpha=0.15,
                        )      
        plt.legend()
        plt.xlabel('Step')
        plt.grid()
        plt.ylabel('Return')
        # plt.title("results")
        # # save the figure
        # plt.savefig(
        #     'eval_trained_plots.png', 
        #     dpi=300, 
        #     bbox_inches='tight',
        #     format='png')
        plt.show()


###########################################
#### scripting #########
###########################################
batch_1 = pd.read_csv('2link_SAC_1.csv')
batch_2 = pd.read_csv('2link_SAC_2.csv')
batch_3 = pd.read_csv('2link_SAC_3.csv')
batch_4 = pd.read_csv('2link_SAC_4.csv')
batch_5 = pd.read_csv('2link_SAC_5.csv')
# batch_6 = pd.read_csv('2link_SAC_6.csv')
# print('batch_1 : \n', batch_1 )
# xxx

order_batch_1 = batch_1[['Step','Value']].sort_values(by='Step')
order_batch_2 = batch_2[['Step','Value']].sort_values(by='Step')
order_batch_3 = batch_3[['Step','Value']].sort_values(by='Step')
order_batch_4 = batch_4[['Step','Value']].sort_values(by='Step')
order_batch_5 = batch_5[['Step','Value']].sort_values(by='Step')

# print('order_batch_1: \n', order_batch_1)
# print('order_batch_2: \n', order_batch_2.shape)
# print('order_batch_3: \n', order_batch_3.shape)
# print('order_batch_4: \n', order_batch_4.shape)
# print('order_batch_5: \n', order_batch_5.shape)
# xxx

###### algorithm limits #################
batch_all = pd.concat([
     order_batch_1['Value'],
     order_batch_2['Value'],
     order_batch_3['Value'],
     order_batch_4['Value'],
     order_batch_5['Value']
     ],axis=1)
stats = batch_all.describe()
overall_mean = np.mean(stats.transpose()['mean'])
overall_max = stats.transpose()['max'].max()

# print('batch_all: \n', batch_all)
# print('stats: \n', stats.transpose())
print('overall_mean: ', overall_mean)
print('overall_max: ', overall_max)
##########################################

run_batch_1 = np.array(running_avg(order_batch_1['Value']))
run_batch_2 = np.array(running_avg(order_batch_2['Value']))
run_batch_3 = np.array(running_avg(order_batch_3['Value']))
run_batch_4 = np.array(running_avg(order_batch_4['Value']))
run_batch_5 = np.array(running_avg(order_batch_5['Value']))
# print('run_batch_1: \n', run_batch_1)


# comparing_multiple_runs
plt.plot(order_batch_1['Step'],run_batch_1[0],color="tomato",alpha=0.8, label="2-link") #std_vs_mean
plt.plot(order_batch_2['Step'],run_batch_2[0],color="moccasin",alpha=0.8, label="2-link") #std_vs_mean
plt.plot(order_batch_3['Step'],run_batch_3[0],color="mediumorchid",alpha=0.8, label="2-link") #std_vs_mean
plt.plot(order_batch_4['Step'],run_batch_4[0],color="teal",alpha=0.8, label="2-link") #std_vs_mean
plt.plot(order_batch_5['Step'],run_batch_5[0],color="chartreuse",alpha=0.8, label="2-link") #std_vs_mean
plt.legend()
plt.show()