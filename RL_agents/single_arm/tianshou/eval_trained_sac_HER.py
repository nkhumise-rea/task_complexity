import gym
import matplotlib.pyplot as plt
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
from tianshou.data import Collector, ReplayBuffer
from tianshou.policy import DDPGPolicy, SACPolicy
from tianshou.utils.net.common import Net, get_dict_state_decorator 
from tianshou.utils.net.continuous import ActorProb,Critic
from tianshou.utils import TensorboardLogger
from tianshou.exploration import GaussianNoise
device = "cuda" if torch.cuda.is_available() else "cpu"

sys.path.insert(0, "..")
from envs.task_HER import SingleLink as one_link_arm
         
def test_sac():
    ##Parameters
    hidden_sizes = [256,256]
    num_episodes = 750
    tau = 5e-3
    gamma = 0.99
    alpha = 0.2
    lr = 1e-3
    buffer_size = 1e6

    #evaluation params
    upload = True
    #simulate = False #Collect Data
    simulate = True #Simulate

    if simulate:
        render = 0.1
        head = 0 #head=[0:GUI, 1:DIRECT]
    else:
        render = 0.0
        head = 1 #head=[0:GUI, 1:DIRECT]
    
    ##Environment
    env = one_link_arm(head)

    #env shapes
    state_shape = {
        'observation' : env.observation_space['observation'].shape,
        'achieved_goal' : env.observation_space['achieved_goal'].shape,
        'desired_goal' : env.observation_space['desired_goal'].shape,
        }
    action_shape = env.action_space.shape 
    max_action = env.action_space.high[0]

    #print('state_shape: ',state_shape)
    #print('action_shape: ',action_shape)
    #print('max_action: ',max_action)
    #print('sigma: ',sigma)

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

    ##buffer
    buf = ReplayBuffer(size=buffer_size)

    ##collectors
    train_collector = Collector(policy,env,buf,exploration_noise=True)
    test_collector = Collector(policy,env)
    #prep before training
    train_collector.reset()
    test_collector.reset()
    env.reset()
    buf.reset()

    if upload:
        policy.load_state_dict(torch.load(
            "../torch_models/sac_link_HER.pth", 
            map_location=torch.device('cpu')
            ))

    #Watch performance
    policy.eval()
    env.reset()
    test_collector.reset()
    result = test_collector.collect(n_episode=num_episodes,render=render)
    #print(f'Final reward:{result["rews"].mean()}, length:{result["lens"].mean()}')

    #tensorboard_results
    if not simulate:
        this_dir = dirname(__file__)
        run_file = 'eval_runs/sac_HER'
        log_path = abspath(join(this_dir,'..',run_file))
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