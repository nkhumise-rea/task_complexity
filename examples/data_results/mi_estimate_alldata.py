import os
import argparse
import optuna
import numpy as np
import pandas as pd

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

    '1link100d': -0.1853, #link_arm < >
    '1link170d': -0.1666, #link_arm < >
    '2linkd': -0.5641, #link_arm < >

    '1link170d_Uniform': -0.1666, #link_arm < >
    '1link170d_Units64': -0.1666, #link_arm < >
    '1link170d_Varian2': -0.1666, #link_arm < >
    '1link170d_Xavier': -0.1666, #link_arm < >
    '1link170d_Bias': -0.1666, #link_arm < >
    '1link170d_Xavier_Uniform': -0.1666, #link_arm < >
    '1link170d_Units256_Xavier': -0.1666, #link_arm < >
    '1link170d_Units256': -0.1666, #link_arm < >

    '1link100d_T55': -0.1853, # 
    '1link100d_T45': -0.1853, # 
    '1link100d_T35': -0.1853, # 
    '1link100d_T25': -0.1853, # 
    '1link100d_T15': -0.1853, # 
    

    '1link100s': -0.4, #link_arm < >
    '1link170s': -0.2, #link_arm < >
    '2links': -22.4, #link_arm < >
    
    '2linksHER': -2.2, #link_arm < >
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

    '1link100d': -5.36, #link_arm < >
    '1link170d': -8.74, #link_arm < >
    '2linkd': -9.18, #link_arm < >

    '1link170d_Uniform': -8.74, #link_arm < >
    '1link170d_Units64': -8.74, #link_arm < >
    '1link170d_Varian2': -8.74, #link_arm < >
    '1link170d_Xavier': -8.74, #link_arm < >
    '1link170d_Bias': -8.74, #link_arm < >
    '1link170d_Xavier_Uniform': -8.74, #link_arm < >
    '1link170d_Units256_Xavier': -8.74, #link_arm < >
    '1link170d_Units256': -8.74, #link_arm < >

    '1link100d_T55': -5.36, # 
    '1link100d_T45': -5.36, # 
    '1link100d_T35': -5.36, # 
    '1link100d_T25': -5.36, # 
    '1link100d_T15': -5.36, # 

    '1link100s': -3.61, #link_arm < >
    '1link170s': -4.32, #link_arm < >
    '2links': -46.59, #link_arm < >

    '2linksHER': -19.6, #link_arm < >
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
    reward_type =  'dense' #'sparse'
    length = 170 #100 
    max_torque = 55 #25 #35 #45 #55
    sampling_type =   'Units256' #'Bias' #'0' #'Uniform' #'Units64' #'Varian2' #'Xavier' #'Bias' #'Xavier_Uniform' #'Units256_Xavier'

    parser = argparse.ArgumentParser()
    parser.add_argument("--n_trials", type=int, default=200, help="n_trials for optuna")
    parser.add_argument("--n_bins", type=int, default=100000, help="number of bins")
    parser.add_argument("--algo_max", action="store_true", help="max(r^algo, r^rand)")
    parser.add_argument("--clip_persent", type=float, default=0.0, help="top/bottom x percent clipping")
    # parser.add_argument("--sourse_path", type=str, default='./CartPole-v0.npy')
    # parser.add_argument("--root_dir", type=str, default='./results/')
    # parser.add_argument("--env", type=str, default='CartPole-v0')

    if link_type == 1 and reward_type == 'dense' and length == 100: 
        if max_torque == 15: env_name = '1link100d_T15'
        elif max_torque == 25: env_name = '1link100d_T25'
        elif max_torque == 35: env_name = '1link100d_T35'
        elif max_torque == 45: env_name = '1link100d_T45'
        elif max_torque == 55: env_name = '1link100d_T55'
        else: env_name = '1link100d'
    elif link_type == 1 and reward_type == 'dense' and length == 170: 
        if sampling_type == 'Uniform': env_name = '1link170d_Uniform'
        elif sampling_type == 'Units64': env_name = '1link170d_Units64'
        elif sampling_type == 'Varian2': env_name = '1link170d_Varian2'
        elif sampling_type == 'Xavier': env_name = '1link170d_Xavier'
        elif sampling_type == 'Bias': env_name = '1link170d_Bias'
        elif sampling_type == 'Xavier_Uniform': env_name = '1link170d_Xavier_Uniform'
        elif sampling_type == 'Units256_Xavier': env_name = '1link170d_Units256_Xavier'
        elif sampling_type == 'Units256': env_name = '1link170d_Units256'
        else: env_name = '1link170d'
    elif link_type == 2 and reward_type == 'dense': env_name = '2linkd'
    elif link_type == 1 and reward_type == 'sparse' and length == 100: env_name = '1link100s'
    elif link_type == 1 and reward_type == 'sparse' and length == 170: env_name = '1link170s'
    elif link_type == 2 and reward_type == 'sparse': env_name = '2links'
    else: 
        print("There is no matching task environment")
        xxx
    args = parser.parse_args()

    print('env_name: ', env_name)
    # xxx

    parser.add_argument("--env", type=str, default=env_name)
    if link_type == 1: 
        # parser.add_argument("--sourse_path", type=str, default='/{}_link_{}_{}_dataset_total.npy'.format(link_type,reward_type,length))
        path = '/{}_link_{}_{}_dataset_total.npy'.format(link_type,reward_type,length)
        if length == 100 and max_torque != 0:
            # print('what1')
            # parser.add_argument("--sourse_path", type=str, default='/{}_link_{}_{}_dataset_total_T{}.npy'.format(link_type,reward_type,length,max_torque))
            path = '/{}_link_{}_{}_dataset_total_T{}.npy'.format(link_type,reward_type,length,max_torque)
            # print('what')
            # xxx
        elif length == 170 and sampling_type != '0':
            path = '/{}_link_{}_{}_dataset_total_{}.npy'.format(link_type,reward_type,length,sampling_type)
    else:
        # parser.add_argument("--sourse_path", type=str, default='/{}_link_{}_dataset_total.npy'.format(link_type,reward_type))
        path = '/{}_link_{}_dataset_total.npy'.format(link_type,reward_type)

    parser.add_argument("--sourse_path", type=str, default=path)
    # print('path: ', path)
    # zzz
    # ccc
    parser.add_argument("--root_dir", type=str, default='./')


    
    args = parser.parse_args()

    # print(args.sourse_path)
    # print(args.root_dir)
    # xxxx

    # save dir
    output_dir = os.path.join(
        args.root_dir, 'normalised/'
        'n_trials{}_clip_persent{}_Initialise'.format(args.n_trials, args.clip_persent),
        )
    os.makedirs(output_dir, exist_ok=True)

    with open(os.path.join(output_dir, "{}_metrics_batch.txt".format(args.env)), "w") as f:
        print("\t".join(_basic_columns), file=f)
    # with open(os.path.join(output_dir, "{}_tables_batch.txt".format(args.env)), "w") as f:
    #     print(" & ".join(_basic_columns), file=f)



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
    print('len: ', all_scores_per_param.shape[0])
    # xxx

    ####################### Normalization of Returns ##############################
    ##### Reward Scaling Test (PIC and POIC are not affected)
    
    # print(all_scores_per_param.shape)
    # df = pd.DataFrame(all_scores_per_param)
    # # print('df: \n ', df)
    # global_min = df.min().min()
    # global_max = df.max().max()
    # # print('global_min: \n ', global_min)
    # # print('global_max: \n ', global_max)

    # df_scaled = (df - global_min)/(global_max - global_min)
    # # print('df_scaled: \n ', df_scaled)
    
    # crown = df_scaled.to_numpy()
    # # print('crown: ', crown.shape)
    # # xxx   

    # all_scores_per_param = crown #scaled_values


    ####################### Normalization of Returns ##############################


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

    with open(os.path.join(output_dir,"{}_metrics_batch.txt".format(args.env)), "a+") as f:
        print("\t".join(str(x) for x in values)+"\n", file=f)
    # with open(os.path.join(output_dir, "{}_tables_batch.txt".format(args.env)), "a+") as f:
    #     print(" & ".join(str(x) for x in values), file=f)

    print('done')
    print('len: ', all_scores_per_param.shape[0])


if __name__ == "__main__":
    main()



# python mi_estimate.py --sourse_path ./results/CartPole-v0.npy --env CartPole-v0

#execution_1: python mi_estimate.py --env one_link
#execution_2: python mi_estimate.py --env two_link