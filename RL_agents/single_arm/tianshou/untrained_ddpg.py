import gym
import matplotlib.pyplot as plt
import time
import os
import numpy as np
import torch
from numpy import random
import argparse

from torch.utils.tensorboard import SummaryWriter

from tianshou.data import Batch, Collector, ReplayBuffer ,VectorReplayBuffer
from tianshou.policy import SACPolicy, DDPGPolicy
from tianshou.trainer import offpolicy_trainer, OffpolicyTrainer, test_episode
from tianshou.utils.net.common import Net
from tianshou.utils.net.continuous import ActorProb, Critic, Actor
from tianshou.utils import TensorboardLogger
from tianshou.env import DummyVectorEnv
from tianshou.exploration import GaussianNoise, OUNoise
device = "cuda" if torch.cuda.is_available() else "cpu"
 
#from link1_task import SingleLink as one_link_arm
from envs.task import SingleLink as one_link_arm

##Parameters
num_envs = 1
hidden_sizes = [256,256]
num_episodes = 500
steps_per_episode = 500
tau = 5e-3
gamma = 0.99
lr_actor = 1e-4
lr_critic = 1e-3
steps = 0
batch_size = 128 #64
steps = 0
exploration_noise = 0.2
buffer_size = 1e6
buffer = ReplayBuffer(size=buffer_size)
buf = VectorReplayBuffer(total_size=buffer_size,buffer_num=num_envs)

##Environment
head = 1
env = one_link_arm(head)
envs = DummyVectorEnv([lambda: env for _ in range(num_envs)])

"""
#vectorized envs
state_shape = envs.observation_space[0].shape
action_shape = envs.action_space[0].shape 
max_action = envs.action_space[0].high[0]
obs = envs.reset()
"""

#plain env
state_shape = env.observation_space.shape
action_shape = env.action_space.shape 
max_action = env.action_space.high[0]
obs = env.reset()

##model
#--preprocessing networks
net_a = Net(
    state_shape,
    hidden_sizes=hidden_sizes, 
    device=device)

net_c = Net(
    state_shape,
    action_shape, 
    hidden_sizes=hidden_sizes,
    concat = True, 
    device=device)

#--Actor & Critics networks
actor = Actor( #deterministic actor
    net_a,
    action_shape,
    max_action = max_action,
    device = device).to(device)

critic = Critic(net_c,device = device).to(device)
actor_optim = torch.optim.Adam(actor.parameters(),lr=lr_actor)
critic_optim = torch.optim.Adam(critic.parameters(),lr=lr_critic)

policy = DDPGPolicy(
    actor,
    actor_optim,
    critic,
    critic_optim,
    tau = tau,
    gamma = gamma,
    exploration_noise = GaussianNoise(sigma=0.5) ,#OUNoise(),
    )

start_time = time.time()

##collectors
train_collector = Collector(policy,envs,buf,exploration_noise=True)
test_collector = Collector(policy,envs)
some_collector = Collector(policy,envs)
#prep before training
train_collector.reset()
test_collector.reset()
envs.reset()
buf.reset()
env.reset()
buffer.reset()

#"""
##Training loop_3
log_path = os.path.join("untrained/runs/ddpg")
writer = SummaryWriter(log_path) 
train_interval = 1
test_interval = 1
update_interval = 1

##logger
#now = datetime.datetime.now().strftime("%y%m%d-%H%M%S")   
logger = TensorboardLogger(
                            writer,
                            train_interval=train_interval,
                            test_interval=test_interval, 
                            )

#for steps in range(500): 
for steps in range(num_episodes*steps_per_episode):
    act = policy(Batch(obs=obs[np.newaxis, :], info={})).act.item()
    obs_next, rew, done, info = env.step(act)
    if done: truncated,terminated = True, True
    else: truncated,terminated = False, False
    buffer.add(Batch(obs=obs, act=act, rew=rew, done=done, obs_next=obs_next, info=info, terminated=terminated, truncated=truncated))
    obs_next = obs

#print(buffer)
act_collection = buffer.act
mean = np.mean( act_collection)
std = np.std( act_collection)
#print('act_col: ',act_collection)
print('mean: ', mean)
print('std: ', std)

#values = np.arange(0,100,1)
#print('values: ', values)
##store data in histogram
log_path = "/home/luca/Documents/PyBullet/single_arm/collections/ddpg"
writer = SummaryWriter(log_path)
writer.add_histogram(
    tag = "ddpg_action/collection",
    values = act_collection,
    global_step = 0,
    )

fig = plt.figure()
plt.hist(act_collection, bins='auto' )
plt.title("matplotlib Histogram w/ 100 bins")
writer.add_figure(
    tag = "ddpg_action/collection/plt_histogram",
    figure = fig,
    global_step = 0,
    )
writer.close()
