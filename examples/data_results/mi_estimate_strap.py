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
    '1link100d': -0.552E-3, #link_arm < >
    '1link170d': -0.552E-3, #link_arm < >
    '2linkd': -46.18E-3, #link_arm < >

    '1link100s': 0.0, #link_arm < >
    '1link170s': 0.0, #link_arm < >
    '2links': 0.0, #link_arm < >
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

    '1link100s': -4.291, #link_arm < >
    '1link170s': -5.012, #link_arm < >
    '2links': -23.89, #link_arm < >
}


def main(data, link_type = 2,reward_type = 'dense',length = 100):
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

    # link_type = 2 #
    # reward_type = 'sparse' #'dense'
    # length = 170 #100 

    parser = argparse.ArgumentParser()
    parser.add_argument("--n_trials", type=int, default=200, help="n_trials for optuna")
    parser.add_argument("--n_bins", type=int, default=100000, help="number of bins")
    parser.add_argument("--algo_max", action="store_true", help="max(r^algo, r^rand)")
    parser.add_argument("--clip_persent", type=float, default=0.0, help="top/bottom x percent clipping")
    
    if link_type == 1 and reward_type == 'dense' and length == 100: env_name = '1link100d'
    elif link_type == 1 and reward_type == 'dense' and length == 170: env_name = '1link170d'
    elif link_type == 2 and reward_type == 'dense': env_name = '2linkd'
    elif link_type == 1 and reward_type == 'sparse' and length == 100: env_name = '1link100s'
    elif link_type == 1 and reward_type == 'sparse' and length == 170: env_name = '1link170s'
    elif link_type == 2 and reward_type == 'sparse': env_name = '2links'
    else: 
        print("There is no matching task environment")
        xxx
    
    parser.add_argument("--env", type=str, default=env_name)
    if link_type == 1: 
        parser.add_argument("--sourse_path", type=str, default='/{}_link_{}_{}_dataset_total.npy'.format(link_type,reward_type,length))
    else:
        parser.add_argument("--sourse_path", type=str, default='/{}_link_{}_dataset_total.npy'.format(link_type,reward_type))
    args = parser.parse_args()

    all_scores_per_param = data
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

    # with open(os.path.join(output_dir, "{}_metrics_batch.txt".format(args.env)), "a+") as f:
    #     print("\t".join(str(x) for x in values)+"\n", file=f)
    # with open(os.path.join(output_dir, "{}_tables_batch.txt".format(args.env)), "a+") as f:
    #     print(" & ".join(str(x) for x in values), file=f)

    # print("\n")
    # print('PIC: ', mi_r)
    # print('POIC: ', mi_o)

    print("\n")
    print('file: ', args.sourse_path)
    return mi_r, mi_o


if __name__ == "__main__":
    ok = main()
    print(ok)



# python mi_estimate.py --sourse_path ./results/CartPole-v0.npy --env CartPole-v0

#execution_1: python mi_estimate.py --env one_link
#execution_2: python mi_estimate.py --env two_link