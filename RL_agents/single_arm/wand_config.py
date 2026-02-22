import wandb

##--sac:

config_name = dict(
    hidden_sizes = [256,256],    
    tau = 5e-3,
    gamma = 0.99,
    alpha = 0.2,
    lr = 1e-3,
    batch_size = 64,
    buffer_size = 1e6,
    epochs = 10,
    steps_per_epoch = 1000,
    steps_per_episode = 500,
    num_episodes = int(10*1000/500),
    step_per_collect = 1,
    episode_per_test = 1,
    )
    
api = wandb.Api()
for num in range(21):
    run_id = "rea/link1/"+str(num)    
    run  = api.run(run_id)
    run.config = config_name
    run.update()

"""
##--ddpg
config_name = dict(
    hidden_sizes = [256,256],
    tau = 5e-3,
    gamma = 0.99,
    lr_actor = 1e-3,
    lr_crtic = 1e-4,
    batch_size = 128,
    buffer_size = 1e6,
    epochs = 10,
    steps_per_epoch = 2000,
    steps_per_episode = 500,
    num_episodes = int(10*2000/500),
    step_per_collect = 1,
    episode_per_test = 1,
    )

api = wandb.Api()
for num in range(21):
    run_id = "rea/link1_ddpg/"+str(num)    
    run  = api.run(run_id)
    run.config = config_name
    run.update()
"""