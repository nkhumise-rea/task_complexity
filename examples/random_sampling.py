import argparse
import os
import numpy as np

from pic.algos import NumpyAgent
from pic.sampler import Sampler, make_env

import sys #link_arm <>
sys.path.insert(0, "..") #link_arm <>
from tasks.single_arm.task import SingleLink as one_link_arm #link_arm <>
from tasks.double_arm.task import DoubleLink as two_link_arm #link_arm <>

def main():
    parser = argparse.ArgumentParser()
    # parser.add_argument("--env", type=str, default="CartPole-v0", help="Open AI gym environments")
    parser.add_argument("--env", type=str, default="one_link", help="Link environments")
    parser.add_argument("--n_units", type=int, default=64, help="number of hidden units")
    parser.add_argument("--n_layers", type=int, default=2, help="number of hidden layers")
    parser.add_argument("--use_bias", action="store_true", help="use bias in NN")
    parser.add_argument("--n_samples", type=int, default=10**4, help="number of parameters sampled from p(\theta)")
    parser.add_argument("--n_episodes", type=int, default=1000, help="number of episode running with parameter \theta")
    parser.add_argument("--root_dir", type=str, default='./results/', help="Root dir to save results")
    parser.add_argument("--random_dist", type=str, choices=['normal', 'uniform', 'xavier_uniform', 'xavier_normal'], default='normal', help="prior distribution of p(\theta)")
    parser.add_argument("--normal_mean", type=float, default=0.0, help="The mean of prior distribution")
    parser.add_argument("--normal_sigma", type=float, default=1.0, help="The sigma of prior distribution")
    parser.add_argument("--uniform_bound", type=float, default=1.0, help="The bound of prior distribution")
    parser.add_argument("--multiprocess", type=int, default=0, help="number of prosess for distrbuted experiments")
    parser.add_argument("--count", type=int, default=0, help="experiment counter")
    args = parser.parse_args()

    #link_arm < Environment >
    head = 1
    max_episode_steps = 50 #link_arm < sample_env.spec.max_episode_steps >
    # print(args.env)
    # ==== environment-selection =====
    # sample_env = link_arm(head) #link_arm < sample_env = make_env(args.env, seed=None) >
    if args.env == "one_link": sample_env = one_link_arm(head)
    else: sample_env = two_link_arm(head)

    agent = NumpyAgent(
        env=sample_env,
        n_hidden_layers=args.n_layers,
        n_hidden_units=args.n_units,
        random_dist=args.random_dist,
        normal_mean=args.normal_mean,
        normal_sigma=args.normal_sigma,
        uniform_bound=args.uniform_bound,
        use_bias=args.use_bias,
        env_name=args.env,
        )

    # save dir
    output_dir = os.path.join(
        args.root_dir,
        'dist_{}_layers{}_units{}'.format(args.random_dist, args.n_layers, args.n_units) #dense_rewards
        )
    os.makedirs(output_dir, exist_ok=True)
    # print(os.path.join(output_dir, "{}_{}.npy".format(args.env, args.count)))

    sampler = Sampler(
        args.env,
        agent,
        max_episode_steps,
        n_samples=args.n_samples,
        n_episodes=args.n_episodes,
        multiprocess=args.multiprocess,
        )
    all_scores_per_param = sampler.sample()
    np.save(os.path.join(output_dir, "{}_{}.npy".format(args.env, args.count)), all_scores_per_param)


if __name__ == "__main__":
    main()


# python random_sampling.py --env CartPole-v0 --random_dist normal --multiprocess 64 --n_units 64 --n_layers 2 --n_samples 1000 --n_episodes 1000

#testing_1:  python random_sampling.py --env one_link --n_units 2 --n_samples 2 --n_episodes 2
#testing_2:  python random_sampling.py --env two_link --n_units 2 --n_samples 2 --n_episodes 2
#local_execution_1:  python random_sampling.py --env one_link --multiprocess 12 --n_samples 10000 --n_episodes 500
#local_execution_2:  python random_sampling.py --env two_link --multiprocess 12 --n_samples 10000 --n_episodes 500