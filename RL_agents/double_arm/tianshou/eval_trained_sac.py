import gym
import matplotlib.pyplot as plt
import time
import os
import numpy as np
import torch
from numpy import random
import argparse
import sys
from os.path import dirname, abspath, join
from torch.utils.tensorboard import SummaryWriter
from tianshou.data import Collector,VectorReplayBuffer,ReplayBuffer
from tianshou.policy import DDPGPolicy, SACPolicy
from tianshou.trainer import offpolicy_trainer
from tianshou.utils.net.common import Net
from tianshou.utils.net.continuous import ActorProb, Actor,Critic
from tianshou.utils import TensorboardLogger
from tianshou.env import DummyVectorEnv
from tianshou.exploration import GaussianNoise
# device = "cuda" if torch.cuda.is_available() else "cpu"
device = "cpu"

sys.path.insert(0, "..")
from envs.task import DoubleLink as two_link_arm

def roll_mean(array,window):
    #print('roll_mean')
    i = 0
    moving_avgs, moving_stds = [], []
    while i < len(array) - window + 1:
        window_avg = np.mean( array[i:i+window] )
        window_std = np.std( array[i:i+window] )
        moving_avgs.append(window_avg)
        moving_stds.append(window_std)
        i += 1
    return np.array(moving_avgs), np.array(moving_stds)

def convert_data(cum_rewards):
    #print('working')
    mean, std = roll_mean(cum_rewards,10)
    lower_bound = mean - std
    upper_bound = mean + std
    return lower_bound, upper_bound, mean 

def plotting(cum_rewards):
        ## Plots
        #fig = plt.figure(figsize=[17,15])
        plt.figure(figsize=[17,15])
        lower_bound, upper_bound, mean = convert_data(cum_rewards)
        episodes = np.arange(0,lower_bound.shape[0],1)

        plt.plot(mean,color="tomato",label='agent'  )
        plt.fill_between(episodes,
                        lower_bound,
                        upper_bound,
                        facecolor="tomato", 
                        alpha=0.15,
                        )      
        plt.legend()
        plt.ylabel('running_score')
        plt.grid()
        plt.xlabel('episodes')
        plt.title("results")
        # save the figure
        plt.savefig(
            'eval_trained_plots.png', 
            dpi=300, 
            bbox_inches='tight',
            format='png')
        plt.show()
          
def test_sac():
    ##Parameters
    num_envs = 1
    hidden_sizes = [256,256]
    num_episodes = 750
    tau = 5e-3
    gamma = 0.99
    alpha = 0.2
    lr = 1e-3
    buf = VectorReplayBuffer(total_size=1e6,buffer_num=num_envs)
    
    #evaluation params
    upload = True
    #simulate = False #Collect Data
    simulate = True #Simulate

    if simulate:
        render = 0.1 #0.1
        head = 0 #head=[0:GUI, 1:DIRECT]
    else:
        render = 0.0
        head = 1 #head=[0:GUI, 1:DIRECT]
    
    ##Environment
    env = two_link_arm(head)
    envs = DummyVectorEnv([lambda: env for _ in range(num_envs)])

    #vectorized envs
    state_shape = envs.observation_space[0].shape
    action_shape = envs.action_space[0].shape 
    max_action = envs.action_space[0].high[0]
    obs = envs.reset()

    ##model
    #--preprocessing networks
    net_a = Net(
        state_shape,
        hidden_sizes=hidden_sizes, 
        device=device)

    net_c1 = Net(
        state_shape,
        action_shape, 
        hidden_sizes=hidden_sizes,
        concat = True, 
        device=device)

    net_c2 = Net(
        state_shape,
        action_shape, 
        hidden_sizes=hidden_sizes,
        concat = True, 
        device=device)

    #--Actor, 2 x Critics networks
    actor = ActorProb(
        net_a,
        action_shape,
        max_action = max_action,
        device = device,
        unbounded = True,
        conditioned_sigma = True).to(device)

    critic1 = Critic(net_c1,device = device).to(device)
    critic2 = Critic(net_c2,device = device).to(device)
    actor_optim = torch.optim.Adam(actor.parameters(),lr=lr)
    critic1_optim = torch.optim.Adam(critic1.parameters(),lr=lr)
    critic2_optim = torch.optim.Adam(critic2.parameters(),lr=lr)

    policy = SACPolicy(
        actor,
        actor_optim,
        critic1,
        critic1_optim,
        critic2,
        critic2_optim,
        tau = tau,
        gamma = gamma,
        alpha = alpha,
        )

    ##collectors
    train_collector = Collector(policy,envs,buf,exploration_noise=True)
    test_collector = Collector(policy,envs)
    some_collector = Collector(policy,envs)
    #prep before training
    train_collector.reset()
    test_collector.reset()
    envs.reset()
    buf.reset()

    if upload:
        policy.load_state_dict(torch.load("../torch_models/sac_link.pth", map_location=torch.device('cpu')))

    #Watch performance
    policy.eval()
    env.reset()
    test_collector.reset()
    result = test_collector.collect(n_episode=num_episodes,render=render)
    #print(f'Final reward:{result["rews"].mean()}, length:{result["lens"].mean()}')

    #tensorboard_results
    if not simulate:
        this_dir = dirname(__file__)
        run_file = 'eval_runs/sac'
        log_path = abspath(join(this_dir,'..',run_file))
        #log_path = "/home/luca/Documents/PyBullet/single_arm/eval_runs"
        writer = SummaryWriter(log_path)
        step = 0
        for i in result["rews"]:
            step +=1
            writer.add_scalar(
                tag = "sac/evaluation/rewards",
                scalar_value = i,
                global_step = step,
                )
        writer.close()

    return result["rews"]

if __name__ == "__main__":
    cum_rewards = test_sac()
    #plotting(cum_rewards)