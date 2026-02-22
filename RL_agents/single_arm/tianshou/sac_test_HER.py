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
from tianshou.data import Collector, HERReplayBuffer
from tianshou.policy import SACPolicy
from tianshou.trainer import offpolicy_trainer
from tianshou.utils.net.common import Net, get_dict_state_decorator
from tianshou.utils.net.continuous import ActorProb, Critic
from tianshou.utils import TensorboardLogger
device = "cuda" if torch.cuda.is_available() else "cpu"
 
sys.path.insert(0, "..")
from envs.task_HER import SingleLink as one_link_arm

##Parameters
num_envs = 1
hidden_sizes = [256,256]
tau = 5e-3
gamma = 0.99
alpha = 0.2
lr = 1e-3
batch_size = 64
buffer_size = 1e6
num_steps_collect = 25000
seed = 0

##Environment
head = 1
env = one_link_arm(head)

#seed
#np.random.seed(seed)
#torch.manual_seed(seed)

#env shapes
state_shape = {
    'observation' : env.observation_space['observation'].shape,
    'achieved_goal' : env.observation_space['achieved_goal'].shape,
    'desired_goal' : env.observation_space['desired_goal'].shape,
    }
action_shape = env.action_space.shape 
max_action = env.action_space.high[0]
obs = env.reset()

dict_state_dec, flat_state_shape = get_dict_state_decorator (
    state_shape = state_shape,
    keys = ['observation','achieved_goal','desired_goal'],
    )

##model
#--preprocessing networks
net_a = dict_state_dec(Net)(
    flat_state_shape,
    hidden_sizes=hidden_sizes, 
    device=device)

net_c1 = dict_state_dec(Net)(
    flat_state_shape,
    action_shape, 
    hidden_sizes=hidden_sizes,
    concat = True, 
    device=device)

net_c2 = dict_state_dec(Net)(
    flat_state_shape,
    action_shape, 
    hidden_sizes=hidden_sizes,
    concat = True, 
    device=device)

#--Actor, 2 x Critics networks
actor = dict_state_dec(ActorProb)(
    net_a,
    action_shape,
    max_action = max_action,
    device = device,
    unbounded = True,
    conditioned_sigma = True).to(device)

critic1 = dict_state_dec(Critic)(net_c1,device = device).to(device)
critic2 = dict_state_dec(Critic)(net_c2,device = device).to(device)
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

##buffer
num_step_episode = 50 #No. steps per episode
def compute_reward_fn(achieved_goal, desired_goal):
    return env.reward(achieved_goal, desired_goal)

buf = HERReplayBuffer(
                    size = buffer_size,
                    compute_reward_fn = compute_reward_fn,
                    horizon = num_step_episode,
                    future_k = 8, 
                      )

##collectors
train_collector = Collector(policy,env,buf,exploration_noise=True)
test_collector = Collector(policy,env)
train_collector.collect(
    n_step = num_steps_collect,
    random = True,
    )
#prep before training
train_collector.reset()
test_collector.reset()
env.reset()
buf.reset()

#"""
##Training loop_3
log_path = os.path.join("../runs/sac_HER")
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
    torch.save(policy.state_dict(),"../torch_models/sac_link_HER.pth")

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

