import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import sys #link_arm <>
sys.path.insert(0, "..") #link_arm <>

#modules
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

def roll_mean(array,window=5):
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
native = False #SAC+HER or SAC-Native
if native == True:
    batch_1 = pd.read_csv('2link_SAC0.csv')
else:
    batch_1 = pd.read_csv('2link_SAC+HER0.csv')

print('batch_1 : \n', batch_1 )
# xxx

order_batch_1 = batch_1[['Step','Value']].sort_values(by='Step')

# print('order_batch_1: \n', order_batch_1)
# print('order_batch_2: \n', order_batch_2.shape)
# print('order_batch_3: \n', order_batch_3.shape)
# print('order_batch_4: \n', order_batch_4.shape)
# print('order_batch_5: \n', order_batch_5.shape)
# xxx

###### algorithm limits #################
batch_all = pd.concat([
     order_batch_1['Value'],
     ],axis=1)
stats = batch_all.describe()
overall_mean = np.mean(stats.transpose()['mean'])
overall_max = stats.transpose()['max'].max()

# print('batch_all: \n', batch_all)
# print('stats: \n', stats.transpose())
print('overall_mean: ', overall_mean)
print('overall_max: ', overall_max)
##########################################