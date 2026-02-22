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
from tianshou.policy import DDPGPolicy
from tianshou.trainer import offpolicy_trainer
from tianshou.utils.net.common import Net, get_dict_state_decorator
from tianshou.utils.net.continuous import Actor, Critic
from tianshou.utils import TensorboardLogger
from tianshou.exploration import GaussianNoise, OUNoise
device = "cuda" if torch.cuda.is_available() else "cpu"

sys.path.insert(0, "..")
from envs.task_HER import DoubleLink as two_link_arm

##Parameters
hidden_sizes = [256,256]
tau = 5e-3
gamma = 0.99
lr_actor = 1e-3
lr_critic = 1e-3
batch_size = 256 #128 #64
exploration_noise = 0.1
buffer_size = 1e6
num_steps_collect = 25000
seed = 0

##Environment
head = 1
env = two_link_arm(head)

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
sigma = max_action*exploration_noise

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

net_c = dict_state_dec(Net)(
    flat_state_shape,
    action_shape, 
    hidden_sizes=hidden_sizes,
    concat = True, 
    device=device)

#--Actor & Critics networks
actor = dict_state_dec(Actor)( #deterministic actor
    net_a,
    action_shape,
    max_action = max_action,
    device = device).to(device)

critic = dict_state_dec(Critic)(net_c,device = device).to(device)
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


##Tensorboard
log_path = os.path.join("../runs/ddpg_HER")
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
    torch.save(policy.state_dict(),"../torch_models/ddpg_link_HER.pth")

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
        step_per_epoch = 10000, 
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
