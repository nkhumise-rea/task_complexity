import gym
import matplotlib.pyplot as plt
import time
import os
import numpy as np
import torch
from numpy import random
from torch.utils.tensorboard import SummaryWriter

from tianshou.data import Batch, Collector, ReplayBuffer
from tianshou.policy import SACPolicy
from tianshou.trainer import offpolicy_trainer
from tianshou.utils.net.common import Net
from tianshou.utils.net.continuous import ActorProb, Critic
from tianshou.utils import TensorboardLogger
from tianshou.env import DummyVectorEnv
device = "cuda" if torch.cuda.is_available() else "cpu"

from task_db import DoubleLink as two_link_arm

#modules
def roll_mean(array,window):
    print('roll_mean')
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

def plot_comp(cum_rewards, random_cum_rewards):
    plt.figure(figsize=[17,15])
    lower_bound, upper_bound, mean = convert_data(cum_rewards)
    ran_lower_bound, ran_upper_bound, ran_mean = convert_data(random_cum_rewards)
    episodes = np.arange(0,lower_bound.shape[0],1)

    plt.plot(mean,color="tomato",label='agent'  )
    plt.fill_between(episodes,
                     lower_bound,
                     upper_bound,
                     facecolor="tomato", 
                     alpha=0.15,
                     )
    
    plt.plot(ran_mean,color="teal", label='random' )
    plt.fill_between(episodes,
                     ran_lower_bound,
                     ran_upper_bound,
                     facecolor="teal", 
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

##Parameters
num_envs = 1
hidden_sizes = [256,256]
num_episodes = 500
tau = 5e-3
gamma = 0.99
alpha = 0.2
lr = 1e-3
steps = 0
batch_size = 64
initial_exploration = 1e2
steps = 0
total_steps = 100
buf = ReplayBuffer(size=1e6)
pi_losses,q1_losses,q2_losses = [],[],[]
print_every = 10
cum_rewards, random_cum_rewards = [],[]
#print('maxsize: {}, data_length: {}'.format(buf.maxsize, len(buf)))

##Environment
head = 0 #head=[0:GUI, 1:DIRECT]
env = two_link_arm(head) 
envs = DummyVectorEnv([lambda: env for _ in range(num_envs)])

#conventional env
state_shape = env.observation_space.shape
action_shape = env.action_space.shape 
max_action = env.action_space.high

#vectorized envs
state_shape = envs.observation_space[0].shape
action_shape = envs.action_space[0].shape 
max_action = envs.action_space[0].high


obs = env.reset() #original env
#print('obs: ', obs)

obs = envs.reset() #vectorized env
#print('obs: ', obs[0])
#print('obs.shape: ', obs.shape)

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

##upload trained model
policy.load_state_dict(torch.load("sac_double_link.pth", map_location=torch.device('cpu')))

start_time = time.time()

##collectors
train_collector = Collector(policy,envs,buf,exploration_noise=True)
test_collector = Collector(policy,envs)
test_collector.reset()
envs.reset()
buf.reset()

#"""
##Training loop_3
log_path = os.path.join("runs")
writer = SummaryWriter(log_path) 

##logger
#now = datetime.datetime.now().strftime("%y%m%d-%H%M%S")   
logger = TensorboardLogger(writer)

#Watch performance
#"""
policy.eval()
result = test_collector.collect(n_episode=250,render=0.10)
random_result = test_collector.collect(n_episode=250,render=0.0, random=True)
cum_rewards = result["rews"]
random_cum_rewards = random_result["rews"]
#"""

"""
for episode in range(num_episodes):
    policy.eval()
    done = False
    score = 0
    obs = envs.reset()
    #print('obs: ', obs )

    while not done:
        steps += 1
        action = policy(Batch(obs=obs[np.newaxis, :], info={}))
        act = action.act.cpu().detach().numpy()[0]
        #print('act: ', act)

        obs_next, rew, done, info = env.step(act)
        #check = env.step(act)
        #print('check: ', check)
        #print('obs_next: ', obs_next)
        #print('rew: ',reward)
        #print('done: ', done)
        #print('info: ', info)

        score += rew
        obs_next = obs
        #time.sleep(0.1)

    cum_rewards.append(score)            
    if episode % print_every == 0:
        print("Episode: {} | Avg_reward: {}".format(episode,score))
        #print("Episode: {} | Avg_reward: {} | steps: {}".format(episode,score,info['step']))
"""

#plotting(cum_rewards) #trained_agent plot
plot_comp(cum_rewards, random_cum_rewards) #random vs trained agent plots
end_time = time.time()
duration = end_time - start_time
print('duration: ', duration)
