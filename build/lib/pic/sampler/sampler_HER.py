import dm2gym
import gym
import random
import multiprocessing as mp
import numpy as np

import pic

import itertools #link_arm <>
import sys #link_arm <>
sys.path.insert(0, "..") #link_arm <>
from tasks.single_arm.task_HER import SingleLink as one_link_arm #link_arm <>
from tasks.double_arm.task_HER import DoubleLink as two_link_arm #link_arm <>

class Sampler(object):
    def __init__(self, env_name, agent, max_episode_steps, n_samples=10**4, n_episodes=10**3, multiprocess=0):
        self.env_name = env_name
        self.agent = agent
        self.n_samples = n_samples
        self.n_episodes = n_episodes
        self.multiprocess = multiprocess
        self.max_episode_steps = max_episode_steps

    def sample(self):
        all_scores_per_param = []
        if self.multiprocess > 0:
            num_worker = mp.cpu_count() ##link_arm < mp.cpu_count() >
            if self.multiprocess > num_worker:
                self.multiprocess = num_worker
            p = mp.Pool(self.multiprocess)
            print("num_worker: {}/{}".format(self.multiprocess, num_worker))

        for samp_num in range(self.n_samples):
            if samp_num % max(1, self.n_samples // 10) == 0:
                print(f"Sample {samp_num}/{self.n_samples}")
            score_episodes = []
            if self.multiprocess > 0:
                episodes_per_worker = max(1, int(np.ceil(self.n_episodes / self.multiprocess)))
                scores = p.starmap(run_episode_wrapper, [[i, self.env_name, self.agent, self.max_episode_steps, episodes_per_worker] for i in range(self.multiprocess)])
                scores = list(itertools.chain(*scores))[:self.n_episodes]
                assert len(scores) == self.n_episodes, f'{len(scores)} != {self.n_episodes}'
                score_episodes += scores
            else:
                # ==== environment-selection =====
                head = 1
                # env = make_env(args.env, seed=None) 
                if self.env_name == "one_link": env = one_link_arm(head)
                else: env = two_link_arm(head)
                # ================================

                for _ in range(self.n_episodes):
                    score = run_episode(env, self.agent, self.max_episode_steps)
                    score_episodes.append(score)
            all_scores_per_param.append(score_episodes)
            self.agent.init_weights()

        if self.multiprocess > 0:
            p.close()

        return np.array(all_scores_per_param)


def make_env(env_name, seed=None):
    if "dm2gym" in env_name:
        env = gym.make(env_name, environment_kwargs={'flat_observation': True})
    else:
        # ==== environment-selection =====
        head = 1
        # env = make_env(args.env, seed=None) >
        if env_name == "one_link": env = one_link_arm(head)
        else: env = two_link_arm(head)

        # print(env_name)
        # xxx
        # ================================

    if seed is not None:
        env.seed(seed)
        random.seed(seed)
        np.random.seed(seed)
    return env


def run_episode(env, agent, max_episode_steps):
    # print(env.reset())
    obs = env.reset()[0] #link_arm < env.reset() >
    # print(obs)
    # print(obs["observation"])
    score = 0
    steps = 0

    # print(env.step([195,95]))

    done = False
    # print('reset_obs: ', obs)
    while not done:
        # if obs.shape: print(obs)
        # if obs.shape:
        #obs
        action = agent.get_action(
            np.array([0.0, 0.912, 1.234, -1.345, 0.123, 0.0, 9.5, 1.70, 0.169])
        )
        # print(obs)
        # print(obs["observation"])

        # print(action)
        next_obs, r, done, _, _ = env.step(action) #link_arm < obs, r, done, _ = env.step(action) >
        print('r: ', r)
        print('next_obs: ', next_obs)
        score += r
        steps += 1
        obs = next_obs
        # print('score: ', score)
        if steps >= max_episode_steps:
            done = True
        # print('score: ', score)
    return score


def run_episode_wrapper(index, env_name, agent, max_episode_steps, num_episodes):
    # ==== environment-selection =====
    head = 1
    # env = make_env(args.env, seed=None) >
    if env_name == "one_link": env = one_link_arm(head)
    # else: env = two_link_arm(head)
    # print(env_name)
    # ================================
    return [run_episode(env, agent, max_episode_steps) for _ in range(num_episodes)]
