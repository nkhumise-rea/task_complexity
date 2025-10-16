import os
import argparse
import optuna
import numpy as np

ALGORITHM_MAX = {
    'CartPole-v0': 200,
    'Pendulum-v0': -128.6266493,
    'MountainCar-v0': -97.2,
    'MountainCarContinuous-v0': 95.89074929,
    'Acrobot-v1': -64.826,
    'Ant-v2': 6584.2,
    'HalfCheetah-v2': 15266.5,
    'Hopper-v2': 3564.07,
    'Walker2d-v2': 5813,
    'Humanoid-v2': 8264,
    'dm2gym:CheetahRun-v0': 795.0,
    'dm2gym:ReacherEasy-v0': 955.0,
    'dm2gym:Ball_in_cupCatch-v0': 978.2,
    '1link100d': -0.552*10^(-3), #link_arm < >
    '1link170d': -0.552*10^(-3), #link_arm < >
    '2linkd': -46.18*10^(-3), #link_arm < >
}

ALGORITHM_AVG = {
    'CartPole-v0': 194.2,
    'Pendulum-v0': -571.5,
    'MountainCar-v0': -143.1,
    'MountainCarContinuous-v0': 12.9,
    'Acrobot-v1': -162.9,
    'Ant-v2': 2450.782353,
    'HalfCheetah-v2': 6047.226471,
    'Hopper-v2': 2206.747059,
    'Walker2d-v2': 3190.777059,
    'Humanoid-v2': 3880.83,
    'dm2gym:CheetahRun-v0': 441.9663239,
    'dm2gym:ReacherEasy-v0': 600.172,
    'dm2gym:Ball_in_cupCatch-v0': 743.21,
    '1link100d': -5.95, #link_arm < >
    '1link170d': -12.546, #link_arm < >
    '2linkd': -16.597, #link_arm < >
}


def main():
    _basic_columns = (
        "environment",
        "normalized_score_A",
        "normalized_score_R",
        "POIC",
        "optimality_marginal",
        "optimality_conditional",
        "PIC",
        "reward_marginal",
        "reward_conditional",
        "variance",
        "temperatures"
        "r_max",
        "r_min",
        "r_mean",
    )

    link_type = 1 #2
    reward_type = 'dense' #'sparse'
    size = 100 #170 

    parser = argparse.ArgumentParser()
    parser.add_argument("--n_trials", type=int, default=200, help="n_trials for optuna")
    parser.add_argument("--n_bins", type=int, default=100000, help="number of bins")
    parser.add_argument("--algo_max", action="store_true", help="max(r^algo, r^rand)")
    parser.add_argument("--clip_persent", type=float, default=0.0, help="top/bottom x percent clipping")
    # parser.add_argument("--sourse_path", type=str, default='./CartPole-v0.npy')
    # parser.add_argument("--root_dir", type=str, default='./results/')
    # parser.add_argument("--env", type=str, default='CartPole-v0')

    if link_type == 1 and reward_type == 'dense' and size == 100:
        parser.add_argument("--env", type=str, default='1link100d')
    elif link_type == 1 and reward_type == 'dense' and size == 170:
        parser.add_argument("--env", type=str, default='1link170d')
    elif link_type == 2 and reward_type == 'dense':
        parser.add_argument("--env", type=str, default='2linkd')
    else: print("There is no matching task environment")
    args = parser.parse_args()


    parser.add_argument("--sourse_path", type=str, default='/{}_link_{}_dataset__50_total.npy'.format(link_type,reward_type))
    # parser.add_argument("--root_dir", type=str, default='./results/data_1vs2dense/{}'.format(args.env))
    parser.add_argument("--root_dir", type=str, default='./tasks/single_arm/prelim_outcomes'.format(args.env))


    
    args = parser.parse_args()

    # print(args.sourse_path)
    # print(args.root_dir)
    # xxxx

    # save dir
    output_dir = os.path.join(
        args.root_dir,
        'n_trials{}_clip_persent{}'.format(args.n_trials, args.clip_persent),
        )
    os.makedirs(output_dir, exist_ok=True)

    with open(os.path.join(output_dir, "{}_metrics.txt".format(args.env)), "w") as f:
        print("\t".join(_basic_columns), file=f)
    with open(os.path.join(output_dir, "{}_tables.txt".format(args.env)), "w") as f:
        print(" & ".join(_basic_columns), file=f)



    # print("HERE")

    # print(args.sourse_path)
    # print("HERE2")
    # # print(np.load(args.sourse_path))
    # print(args.root_dir)
    # print(args.root_dir+args.sourse_path)
    # way = args.root_dir+args.sourse_path
    # print(way)
    # all_scores_per_param = np.load(way)
    # print(all_scores_per_param.shape)
    # xxx

    # all_scores_per_param = np.load(os.path.join(args.root_dir,'{}'.format(args.sourse_path)))
    way = args.root_dir+args.sourse_path
    all_scores_per_param = np.load(way)
    all_mean_scores = all_scores_per_param.mean(axis=1)

    if args.clip_persent > 0:
        upper = np.percentile(all_mean_scores, 100-args.clip_persent)
        lower = np.percentile(all_mean_scores, args.clip_persent)
        all_scores_per_param = np.clip(all_scores_per_param, lower, upper)

    all_scores = all_scores_per_param.flatten()
    r_max = all_scores.max()
    r_min = all_mean_scores.min()
    r_mean = all_scores.mean()

    variance = 0 if (r_max - r_min) == 0 else all_scores.var()/(r_max - r_min)

    if args.algo_max:
        r_max = max(ALGORITHM_MAX[args.env], r_max)

    def objective(trial):
        temperature = trial.suggest_loguniform('temperature', 1e-4, 2e4)
        p_o1 = np.exp((all_scores-r_max)/temperature).mean()
        p_o1_ts = np.exp((all_scores_per_param-r_max)/temperature).mean(axis=1)
        marginal = -p_o1*np.log(p_o1 + 1e-12) - (1-p_o1)*np.log(1-p_o1 + 1e-12)
        conditional = np.mean(-p_o1_ts*np.log(p_o1_ts + 1e-12) - (1-p_o1_ts)*np.log(1-p_o1_ts + 1e-12))
        mutual_information = marginal - conditional

        return mutual_information

    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=args.n_trials)

    # POIC
    trial = study.best_trial
    mi_o = trial.value
    temperature = trial.params['temperature']
    p_o1 = np.exp((all_scores-r_max)/temperature).mean()
    p_o1_ts = np.exp((all_scores_per_param-r_max)/temperature).mean(axis=1)
    h_o = -p_o1*np.log(p_o1) - (1-p_o1)*np.log(1-p_o1)

    h_o_t = np.mean(-p_o1_ts*np.log(p_o1_ts + 1e-12) - (1-p_o1_ts)*np.log(1-p_o1_ts + 1e-12))

    # PIC
    bins = args.n_bins
    hist = np.histogram(all_scores, bins=args.n_bins)
    discretization_all = hist[0] / len(all_scores)
    entropy_all = - np.sum(discretization_all * np.log(discretization_all + 1e-12))
    discretization_r_theta = [np.histogram(x, bins=hist[1])[0] / len(x) for x in all_scores_per_param]
    entropy_r_theta = - np.mean([np.sum(p_r_theta * np.log(p_r_theta + 1e-12)) for p_r_theta in discretization_r_theta])
    mi_r = entropy_all - entropy_r_theta

    normalized_score_A = (ALGORITHM_AVG[args.env] - r_min) / (max(ALGORITHM_MAX[args.env], r_max) - r_min)
    normalized_score_R = (r_mean - r_min) / (max(ALGORITHM_MAX[args.env], r_max) - r_min)

    # save in scores.txt
    values = (
        args.env,
        normalized_score_A,
        normalized_score_R,
        mi_o,
        h_o,
        h_o_t,
        mi_r,
        entropy_all,
        entropy_r_theta,
        variance,
        temperature,
        r_max,
        r_min,
        r_mean,
        )

    with open(os.path.join(output_dir, "{}_metrics.txt".format(args.env)), "a+") as f:
        print("\t".join(str(x) for x in values)+"\n", file=f)
    with open(os.path.join(output_dir, "{}_tables.txt".format(args.env)), "a+") as f:
        print(" & ".join(str(x) for x in values), file=f)


if __name__ == "__main__":
    main()



# python mi_estimate.py --sourse_path ./results/CartPole-v0.npy --env CartPole-v0

#execution_1: python mi_estimate.py --env one_link
#execution_2: python mi_estimate.py --env two_link