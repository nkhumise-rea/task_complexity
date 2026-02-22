import gym
import matplotlib.pyplot as plt
import time
import os
import numpy as np
import torch
from numpy import random
import argparse
import sys
from torch.utils.tensorboard import SummaryWriter
from tianshou.data import Collector,VectorReplayBuffer,ReplayBuffer
from tianshou.policy import DDPGPolicy
from tianshou.trainer import offpolicy_trainer
from tianshou.utils.net.common import Net
from tianshou.utils.net.continuous import Actor, Critic
from tianshou.utils import TensorboardLogger
from tianshou.env import DummyVectorEnv
from tianshou.exploration import GaussianNoise, OUNoise
device = "cuda" if torch.cuda.is_available() else "cpu"

sys.path.insert(0, "..")
from envs.task import SingleLink as one_link_arm

##Parameters
num_envs = 1
hidden_sizes = [256,256]
tau = 5e-3
gamma = 0.99
lr_actor = 1e-3
lr_critic = 1e-3
batch_size = 256 #128 #64
exploration_noise = 0.1
buffer_size = 1e6
buf = VectorReplayBuffer(total_size=buffer_size,buffer_num=num_envs)
seed = 0

##Environment
head = 1
env = one_link_arm(head)
envs = DummyVectorEnv([lambda: env for _ in range(num_envs)])

#seed
#np.random.seed(seed)
#torch.manual_seed(seed)

#vectorized envs
state_shape = envs.observation_space[0].shape
action_shape = envs.action_space[0].shape 
max_action = envs.action_space[0].high[0]
obs = envs.reset()
sigma = max_action*exploration_noise

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
    exploration_noise = GaussianNoise(sigma=sigma), #OUNoise(),
    estimation_step = 1,
    #reward_normalization = True
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

##Tensorboard
log_path = os.path.join("runs/ddpg")
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

##save model
def save_best_fn(policy):
    torch.save(policy.state_dict(),"../torch_models/ddpg_link.pth")

def stop_fn(mean_rewards):
    if mean_rewards > -4.0:
        return True
    else:
        return False

result_f = offpolicy_trainer(
        policy,
        train_collector,
        test_collector,
        max_epoch = 10,
        step_per_epoch = 5000, 
        step_per_collect = 1,
        episode_per_test = 1,
        batch_size = batch_size,
        save_best_fn = save_best_fn,
        #stop_fn = stop_fn,
        logger = logger,
        update_per_step = 1,
        test_in_train = False,
        show_progress = True,
        )

end_time = time.time()
duration = end_time - start_time
print('duration: ', duration)