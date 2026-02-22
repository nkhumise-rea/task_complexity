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
from tianshou.data import Batch, Collector, ReplayBuffer ,VectorReplayBuffer
from tianshou.policy import SACPolicy
from tianshou.trainer import offpolicy_trainer, OffpolicyTrainer, test_episode
from tianshou.utils.net.common import Net
from tianshou.utils.net.continuous import ActorProb, Critic
from tianshou.utils import TensorboardLogger
from tianshou.env import DummyVectorEnv
# device = "cuda" if torch.cuda.is_available() else "cpu"
device = "cpu"
 
parser = argparse.ArgumentParser()
parser.add_argument("--count", type=int, default=0, help="experiment counter")
args = parser.parse_args()

sys.path.insert(0, "..")
from envs.task_obstacle import DoubleLink as two_link_arm

##Parameters
num_envs = 1
hidden_sizes = [256,256]
tau = 5e-3
gamma = 0.99
alpha = 0.2
lr = 1e-3
batch_size = 64
buffer_size = 1e6
buf = VectorReplayBuffer(total_size=buffer_size,buffer_num=num_envs)

##Environment
head = 1
env = two_link_arm(head)
envs = DummyVectorEnv([lambda: env for _ in range(num_envs)])

#vectorized envs
state_shape = envs.observation_space[0].shape
action_shape = envs.action_space[0].shape 
max_action = envs.action_space[0].high
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

#"""
##Training loop_3
log_path = os.path.join(f"runs/sac_obstacle_{args.count}")
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

def save_best_fn(policy):
    torch.save(policy.state_dict(),f"../torch_models/sac_link_obstacle_{args.count}.pth")

def stop_fn(mean_rewards):
    if mean_rewards > -4.0:
        return True
    else:
        return False

result_f = offpolicy_trainer(
        policy,
        train_collector,
        test_collector,
        max_epoch = 20,
        step_per_epoch = 10000, #5000
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