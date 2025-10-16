import random
from copy import copy, deepcopy
import numpy as np
from collections import namedtuple, deque
import matplotlib.pyplot as plt
import time
import pandas as pd

#
import argparse

#pytorch
import torch
import torch.nn as nn
import torch.optim as optima
import torch.nn.functional as F
from torch.distributions.normal import Normal

"""#module_from_different_director
import sys
sys.path.insert(0, '/home/luca/Documents/PyBullet/single_arm')
from task import SingleLink as one_link_arm
"""
##locate task.py (modules)
import sys
sys.path.insert(0, "..")
from task_1T15 import SingleLink as one_link_arm


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

## Model
class NN(nn.Module): #actor_model
    def __init__(self,num_states,num_actions,num_hidden_l1,num_hidden_l2,act_limit):
        super(NN, self).__init__()

        self.act_limit = act_limit
        self.net =nn.Sequential(
            nn.Linear(num_states,num_hidden_l1,bias=False),
            nn.ReLU(),
            nn.Linear(num_hidden_l1,num_hidden_l2,bias=False),
            nn.ReLU(),
            nn.Linear(num_hidden_l1,num_actions,bias=False),
            nn.Tanh(),
            )

        #initialization
        for m in self.modules():
            if isinstance(m, nn.Linear):
                #nn.init.constant_(m.weight,12)
                #nn.init.uniform_(m.weight, -3e-3, 3e-3 ) 
                nn.init.normal_(m.weight, mean=0.0, std=1.0)
                #nn.init.xavier_normal_(m.weight)
                #m.bias.data.fill_(0.0)
          
    def forward(self, state):
        # print('network_state: ', state)
        b_action = self.net(state) #bound action [-1,1]
        # print('b_action: ', b_action)
        # xxx
        action = b_action*self.act_limit #scale actions
        #print('bound_action: ',b_action)
        #print('action: ',action)
        return action

class RANDOM_AGENT():
    def __init__(self,head=0):
        self.env = one_link_arm(head) #[0:GUI, 1:DIRECT]
        self.num_episodes = 500 #50 #total number of episodes
        self.window_size = 10 #rolling window size
        self.duration = None 
        self.count = 0

    #print_model_weights
    def print_model(self,model):
        #for p in model.parameters(): #w/t names
        #    print(p.data)
        for n,p in model.named_parameters(): #w/ names
            # print(n)
            print(p.data)

    ## Policy 
    def output_action(self,state):
        state = state.to(device)
        action = self.model(state)
        #print('action: ', action)
        return action.detach().cpu().numpy()[0]

    def evaluate(self):
        #start_time = time.time()
        
        ## Configurations
        #hyperameters
        num_episodes = self.num_episodes
        print_every = 10
        steps = 0

        #model_params
        num_states = self.env.observation_space.shape[0]
        num_actions = self.env.action_space.shape[0]
        num_hidden_l1 = 32 #5 
        num_hidden_l2 = 32 #5 
        act_limit = self.env.action_space.high[0]
        # print('act_limit: ',act_limit)

        # print('num_states: ', num_states)
        # print('num_actions: ', num_actions)

        #declare model
        self.model = NN(num_states,
                      num_actions,
                      num_hidden_l1,
                      num_hidden_l2,
                      act_limit).to(device)

        # self.print_model(self.model) #print model_weights
        # print('new_game')

        #"""
        self.episodic_return = []
        reward_array = []
        for episode in range(num_episodes):
            done = False
            steps = 0 
            score = 0
            obs = self.env.reset()
            # print('obs: ', obs)
            # print('obs: ', obs[0])
            # xxx

            # state = torch.tensor(obs[0]).float() 
            state = torch.tensor(obs).float() 
            # print('obs: ', obs)
            # print('state: ', state)

            reward_array = []
            while not done:
            #for _ in range(10):
                steps += 1

                # print('state_inloop', state)
                action = self.output_action(state) #NN_model
                #action = self.env.action_space.sample()[0] #random_agent_without_NN                
                # # Batch(obs=i, act=i, rew=i, terminated=0, truncated=0, obs_next=i + 1, info={})
                # print(self.env.step(action))
                # print(self.env.step(action))
                # print(self.env.step(action))
                # print(self.env.step(action))

                # print('state.shape: ', state)
                # if state.shape:
                #     # print()
                #     xxxx

                next_obs, reward, done, _, _ = self.env.step(action)
                # print('next_obs: ', next_obs)

                # xxx
                reward_array.append(reward)
                score += reward
                #print('distance: ',info['distance'])
                # print('state: ', state)
                #print(next_obs)
                next_state = torch.tensor(next_obs).float() 
                state = next_state
                #time.sleep(.01)
            # print('collection: ', reward_array)
            # print('collection: ', len(reward_array))
            # print('score: ', score)
            # xxxx
            self.episodic_return.append(score)            
            # if episode % print_every == 0:
            #     print("Episode: {} | Avg_reward: {}".format(episode,score))
            #    print("Episode: {} | Avg_reward: {} | steps: {}".format(episode,score,info['step']))
        return self.episodic_return #self.cum_running_score, self.std_running_score

if __name__ == '__main__':  
    agent = RANDOM_AGENT(1) # head=[0:GUI, 1:DIRECT]

    """
    n_returns = agent.evaluate()
    mp = agent.mean_performance(n_returns)
    var = agent.variance_performance(n_returns)
    #data['mean'] = agent.mean_performance(n_returns)
    #data['var'] = agent.variance_performance(n_returns)

    print('n_returns: ',n_returns)
    print('mean: ',mp)
    print('vars: ', var)
    #print(data)
    #data_dic[ data_name[i] ] = data[0]
    #data = [] #empty list for new storage
    #agent.plotting(data_dic)

    
    for _ in range(2):
        n_returns = agent.evaluate()
        mp = agent.mean_performance(n_returns)
        var = agent.variance_performance(n_returns)
        print('n_returns: ',n_returns)
        print('mean: ',mp)
        print('vars: ', var)
    """

    parser = argparse.ArgumentParser()
    parser.add_argument("--count", type=int, default=0, help="experiment counter")
    args = parser.parse_args()

    # print(len(agent.evaluate()))
    # print(args.count)
    # xxxx
    link_type = 1 #2
    reward_type = 'dense' 
    num_samples = 200
    samples = {}

    torque_change = True
    max_torque = 15 #25 #35 #30

    for sample in range(num_samples):
        name = 'sample_'+str(sample)
        print(name)
        #n_returns = agent.evaluate()
        samples[name] = agent.evaluate() #sample_returns
        #print('n_returns: ', n_returns)

    #print(samples)
    samples_df = pd.DataFrame(samples) #returns_per_sample

    # print('samples_df: \n',samples_df)

    stats_df = samples_df.describe() #statistical summary 
    means_df = stats_df.loc['mean'] #means_of_samples
    std_df = stats_df.loc['std']
    # maxs_df = stats_df.loc['max']

    # print('stats_df: \n',stats_df)
    # print('maxs_df: \n',maxs_df)
    # print('samples_df.var(): \n',samples_df.var(ddof=0))

    
    summary_data_df = pd.DataFrame()
    summary_data_df['means'] = means_df
    summary_data_df['stds'] = std_df

    # print('summary_data_df: \n', summary_data_df)
    
    summary_data_df['rank'] = summary_data_df['means'].rank()

    # print('summary_data_df: \n', summary_data_df)

    # print('neat: \n',summary_data_df[['means','stds','rank']].sort_values(by='means'))

    order_summary_data_df = summary_data_df[['means','stds','rank']].sort_values(by='means').reset_index(drop=True) #.to_dict('index') #
    
    # print('order_summary_data_df: \n', order_summary_data_df)


    ordered_means_df = order_summary_data_df['means'].to_numpy() #means_of_samples
    ordered_std_df = order_summary_data_df['stds'].to_numpy()
    ordered_var_df = ordered_std_df**2
    ordered_rank_df = order_summary_data_df['rank'].to_numpy()

    # print('ordered_means_df: \n', ordered_means_df)
    # print('ordered_std_df: \n', ordered_std_df)
    # print('ordered_var_df: \n', ordered_var_df)
    # print('ordered_rank_df: \n', ordered_rank_df)
    # xxx

    trans_samples_df = samples_df.T
    # trans_samples_df['means'] = means_df
    # trans_samples_df['stds'] = std_df
    # trans_samples_df['rank'] = trans_samples_df['means'].rank()
    # order_trans_samples_df = trans_samples_df.sort_values(by='rank')

    # rearrange_df = order_trans_samples_df.rename(dict(zip(order_trans_samples_df.index,order_trans_samples_df['rank']))) #rename rows according to column values
    # sorted_rank_data_df = rearrange_df.drop(['means', 'stds', 'rank'], axis=1) #drop means,stds,rank | 

    #print(samples_df)
    #print(stats_df)
    #print(means_df)
    #print(std_df)
    #print(summary_data_df)
    #print(order_summary_data_df)
    #print(trans_samples_df)
    #print(order_trans_samples_df)
    #print(rearrange_df)
    #print(sorted_rank_data_df)
    
    #save to csv
    # samples_df.to_csv('data_1link_RWG_a',encoding='utf-8',index=False)
    
    # print('trans_samples_df: \n', trans_samples_df.reset_index(drop=True))
    # np.save('one_link_dense',trans_samples_df)
    # np.save('{}_link_{}_dataset_{}'.format(link_type,reward_type,args.count),trans_samples_df)
    if torque_change == True:
        np.save('{}_link_{}_dataset_{}_one_T{}'.format(link_type,reward_type,args.count,max_torque),trans_samples_df)
    else:
        np.save('{}_link_{}_dataset_{}_one'.format(link_type,reward_type,args.count),trans_samples_df)

    # xxx

    ##plot std vs means
    #summary_data_df.plot(kind='scatter', x='means', y='std') #std_vs_mean
    """
    plt.subplot(2,1,1)
    plt.scatter(summary_data_df['means'],summary_data_df['stds']) #std_vs_mean
    
    plt.subplot(2,2,3)
    plt.hist(summary_data_df['means'], log=True) #mean_histogram
    
    plt.subplot(2,2,2)
    plt.plot(sorted_rank_data_df, 'rx')

    plt.subplot(2,2,4)
    #plt.scatter(summary_data_df['rank'],summary_data_df['means']) #mean_vs_rank
    plt.plot(sorted_rank_data_df, 'rx')
    plt.plot(order_summary_data_df['rank'],order_summary_data_df['means'], 'b') #mean_vs_rank
    """

    #order_trans_samples_df.plot(x='rank',y=['means','stds'])
    #plt.scatter(order_summary_data_df['rank'],order_summary_data_df['means'])
    #plt.plot(rearr2, 'o')
    # val_size = 10
    # figsize = [val_size,val_size]
    # plt.figure(figsize=figsize)
    # plt.scatter(summary_data_df['means'],summary_data_df['stds'], color="mediumorchid") #std_vs_mean
    # plt.ylabel('std')
    # plt.xlabel('mean')
    # plt.savefig('std_vs_mean.png', dpi=300, bbox_inches='tight')
    
    # plt.figure(figsize=figsize)
    # plt.hist(summary_data_df['means'], log=True) #mean_histogram
    # plt.xlabel('mean')
    # plt.savefig('mean_hist.png', dpi=300, bbox_inches='tight')
    
    # plt.figure(figsize=figsize)
    # # plt.plot(sorted_rank_data_df, 'rx') #returns_vs_rank
    # plt.plot(order_summary_data_df['rank'],order_summary_data_df['means'], 'b') #mean_vs_rank
    # plt.ylabel('score & mean')
    # plt.xlabel('rank')
    # plt.savefig('plots.png', dpi=300, bbox_inches='tight')
    # plt.show()

    val_size = 3
    f_size = 11
    f_size = 9
    figsize = [3*val_size,val_size]
    plt.figure(figsize=figsize)

    plt.subplot(1,3,3)
    # plt.scatter(summary_data_df['means'],summary_data_df['stds'],"o", color="mediumorchid",alpha=0.2) #std_vs_mean
    plt.scatter(ordered_means_df,ordered_std_df ,color="mediumorchid",alpha=0.1) #std_vs_mean
    # plt.scatter(ordered_means_df,ordered_var_df ,color="mediumorchid",alpha=0.1) #std_vs_mean
    # plt.ylabel('std')
    # plt.xlabel('mean')
    plt.ylabel('$\sqrt{V_{n}}$', fontsize=f_size)
    plt.xlabel('$M_{n}$', fontsize=f_size)
    plt.xlim(-500,0)
    # print('ordered_std_df: ', ordered_std_df)
    # print('ordered_std_df: ', ordered_std_df.max())
    y_max = ordered_std_df.max()+10
    plt.ylim(0,y_max)

    plt.title("{}-link arm ({})".format(link_type,reward_type), fontsize=f_size)
    # plt.savefig('std_vs_mean.png', dpi=300, bbox_inches='tight')
    # plt.show()
    # xxx

    # plt.figure(figsize=figsize)
    plt.subplot(1,3,1)
    plt.hist(ordered_means_df, log=True, edgecolor="black") #mean_histogram
    # plt.xlabel('mean')
    plt.xlabel('$M_{n}$', fontsize=f_size)
    plt.ylabel('log(Counts)', fontsize=f_size)
    plt.xlim(-500,0)
    plt.title("{}-link arm ({})".format(link_type,reward_type), fontsize=f_size)
    # plt.savefig('mean_hist.png', dpi=300, bbox_inches='tight')
    # plt.show()
    # xxx

    # plt.figure(figsize=figsize)
    # plt.plot(order_summary_data_df['rank'],order_summary_data_df['means'], 'b') #mean_vs_rank
    # plt.plot(z2['rank'],z2['mean'], 'rx') #returns_vs_rank

    # z3 = np.array(samples_df[120].sort_values())
    # r3 = np.arange(1,z3.shape[0]+1) 

    # print(r3)
    # print(order_summary_data_df['rank'])


    # testy = pd.DataFrame()
    # testy['mean'] = z3
    # testy['rank'] = r3
    # testy['ordered'] = ordered_means
    # print(testy[:-5])


    plt.subplot(1,3,2)
    # plt.plot(testy['rank'],testy['ordered'], 'k') #mean_vs_rank
    plt.plot(ordered_rank_df,ordered_means_df, 'k') #mean_vs_rank
    # plt.ylabel('score & mean')
    plt.ylabel('$M_{n}$',fontsize=f_size)
    plt.xlabel('$R_{n}$', fontsize=f_size)
    plt.ylim(-500,0)
    plt.title("{}-link arm ({})".format(link_type,reward_type), fontsize=f_size)
    # plt.savefig('plots.png', dpi=300, bbox_inches='tight')


    plt.tight_layout()
    # plt.savefig('{}-link_{}_plots_{}.png'.format(link_type,reward_type,args.count), dpi=300, bbox_inches='tight')
    # plt.savefig('{}-link_{}_plots_{}_one.png'.format(link_type,reward_type,args.count), dpi=300, bbox_inches='tight')
    
    if torque_change == True:
        plt.savefig('{}-link_{}_plots_{}_one_T{}.png'.format(link_type,reward_type,args.count,max_torque), dpi=300, bbox_inches='tight')
        # np.save('{}_link_{}_dataset_{}_one_T{}'.format(link_type,reward_type,args.count,max_torque),trans_samples_df)
    else:
        plt.savefig('{}-link_{}_plots_{}_one.png'.format(link_type,reward_type,args.count), dpi=300, bbox_inches='tight')
        
        # np.save('{}_link_{}_dataset_{}_one'.format(link_type,reward_type,args.count),trans_samples_df)

    # plt.show()



