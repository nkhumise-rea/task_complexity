import random
from copy import copy, deepcopy
import numpy as np
from collections import namedtuple, deque
import matplotlib.pyplot as plt
import time

#pytorch
import torch
import torch.nn as nn
import torch.optim as optima
import torch.nn.functional as F
from torch.distributions.normal import Normal
from torch.utils.tensorboard import SummaryWriter

from envs.task import SingleLink as one_link_arm

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class AGENT():
    def __init__(self,head=0,agent_type=0):
        

        #Input1: {p.GUI=False/0, p.DIRECT=True/1} 
        #Input2:{sparse_rewards=False/0, dense_rewards=True/1}
        #self.env = one_link_arm(head) #see above comment ^
        self.env = one_link_arm(head)
        self.agent_type = agent_type
        self.cum_rewards = []
        self.cum_running_score = []
        self.std_running_score = []
        self.num_episodes = 5000 #total number of episodes
        self.window_size = 10 #rolling window size
        self.epsilon_cum = []
        self.duration = None 
        self.old_error = None

        self.agent = None
        self.random = None
        self.pid = None
        self.count = 0
        self.ss = []


    ### Re-setting all parameters
    def seed_everything(self, seed: int):
        import random
        import numpy as np
        import torch
        
        random.seed()
        np.random.seed()
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
    
    #direction of error
    def error_direction(self,state):
        eta = np.around(state[3] - state[0],2)
        eta_abs = np.around(np.abs(state[3] - state[0]),2)
        #print('e: {}, |e|: {} '.format(eta, eta_abs))
        if eta < 0 and eta_abs < np.pi: flag = True
            #print('cw')
        elif eta > 0 and eta_abs > np.pi: flag = True
            #print('cw')
        elif eta < 0 and eta_abs > np.pi: flag = False
            #print('ccw')
        else: flag = False
        return flag

    ## Policy 
    def output_action(self, state):
        #print('self.agent_type: ',self.agent_type)
        if  self.agent_type == 0: #random_agent
            action = self.env.action_space.sample()[0]
            return action
        else: #p-controller
            kp = 1.0 #-50      
            error = self.env.angle_error(state)
            flag = self.error_direction(state)
            if flag: action = np.tanh(-kp*error) #CW
            else: action = np.tanh(kp*error) #CCW
            return action

    def state_input(self,state):
        state = state.numpy()
        return [state[0], state[3]]

    def evaluate(self):
        start_time = time.time()
        
        ## Configurations
        num_episodes = self.num_episodes
        steps = 0
        print_every = 10
        steps = 0 
        log_path = "/home/luca/Documents/PyBullet/single_arm/runs/random"
        writer = SummaryWriter(log_path)

        for episode in range(num_episodes):
            done = False
            score = 0

            ##issue goal & start positions
            obs = self.env.reset()
            #print('obs: ', obs)

            state = torch.tensor(obs).float() 
            #print('state: ', state)
            
            while not done:
                action = self.output_action(state) #trained_agent
                next_obs, reward, done, info = self.env.step(action)
                score += reward
                next_state = torch.tensor(next_obs).float() 
                state = next_state

                #time.sleep(.01)
                steps += 1

            #tensorboard_results
            writer.add_scalar(
                tag = "random/evaluation/rewards",
                scalar_value = score,
                global_step = steps,
                )
            self.cum_rewards.append(score)
            
            #if episode % print_every == 0:
            #    print("Episode: {} | Avg_reward: {}".format(episode,score))
            #    #print("Episode: {} | Avg_reward: {} | steps: {}".format(episode,score,info['step']))
        
        writer.close()
        end_time = time.time()
        self.duration = end_time - start_time
        return self.cum_rewards 

    #modules
    #"""
    def roll_mean(self,array,window):
        #print('roll_mean')
        i = 0
        moving_avgs, moving_stds = [], []
        while i < len(array) - window + 1:
            window_avg = np.mean( array[i:i+window] )
            window_std = np.std( array[i:i+window] )
            moving_avgs.append(window_avg)
            moving_stds.append(window_std)
            i += 1
        return np.array(moving_avgs), np.array(moving_stds)

    def convert_data(self,cum_rewards):
        #print('working')
        mean, std = self.roll_mean(cum_rewards,10)
        lower_bound = mean - std
        upper_bound = mean + std
        return lower_bound, upper_bound, mean 

    def plotting(self,cum_rewards):
            ## Plots
            #fig = plt.figure(figsize=[17,15])
            plt.figure(figsize=[17,15])
            lower_bound, upper_bound, mean = self.convert_data(cum_rewards)
            episodes = np.arange(0,lower_bound.shape[0],1)

            plt.plot(mean,color="tomato",label='agent'  )
            plt.fill_between(episodes,
                            lower_bound,
                            upper_bound,
                            facecolor="tomato", 
                            alpha=0.15,
                            )      
            plt.legend()
            plt.ylabel('running_score')
            plt.grid()
            plt.xlabel('episodes')
            plt.title("results")
            # save the figure
            plt.savefig(
                'eval_trained_plots.png', 
                dpi=300, 
                bbox_inches='tight',
                format='png')
            plt.show()

    def plot_multiple(self,data_dic):

        random_data, pcon_data = None, None
        data_name = ['random','pcon']

        if data_dic[ data_name[0] ] is not None:
            random_data  = data_dic[ data_name[0] ]

        if data_dic[ data_name[1] ] is not None:
            pcon_data  = data_dic[ data_name[1] ]
             
        ## Plots
        plt.figure(figsize=[17,15])
        ran_lb, ran_ub, ran_mean = self.convert_data(random_data)
        pcon_lb, pcon_ub, pcon_mean = self.convert_data(pcon_data)
        episodes = np.arange(0,ran_lb.shape[0],1)
        
        plt.plot(ran_mean,color="royalblue",label='random' )
        plt.fill_between(episodes,
                        ran_lb,
                        ran_ub,
                        facecolor="royalblue", 
                        alpha=0.15,
                        )
        
        plt.plot(pcon_mean,color="teal", label='p_controller' )
        plt.fill_between(episodes,
                        pcon_lb,
                        pcon_ub,
                        facecolor="teal", 
                        alpha=0.15,
                            )
        
        plt.legend()
        plt.ylabel('running_score')
        plt.grid()
        plt.xlabel('episodes')
        #listOf_Yticks = np.arange(-700.0, 0.0, 50)
        #plt.yticks(listOf_Yticks)
        #plt.ylim(-600.0,0.0)

        # save the figure
        """
        if self.agent_type == 1: #trained
            plt.savefig('eval_trained_plots.png', dpi=300, bbox_inches='tight')
        elif self.agent_type == 0: #random
            plt.savefig('eval_random_plots.png', dpi=300, bbox_inches='tight')
        else: #scripted
            plt.savefig('eval_script_plots.png', dpi=300, bbox_inches='tight')
        """
        plt.show()
    #"""
    """
    def convert_data(self,cum_rewards):
        value = np.lib.stride_tricks.sliding_window_view(cum_rewards,self.window_size)
        cum_running_score = np.mean(value,axis=-1)
        std_running_score = np.std(value,axis=-1)
        
        lower_bound = np.array(cum_running_score) - np.array(std_running_score)
        upper_bound = np.array(cum_running_score) + np.array(std_running_score) 
        return lower_bound, upper_bound, cum_running_score       

    def plot_multiple(self,data_dic):

        random, pid = None, None
        data_name = ['random','pid']

        if data_dic[ data_name[0] ] is not None:
            random  = data_dic[ data_name[0] ]

        if data_dic[ data_name[1] ] is not None:
            pid  = data_dic[ data_name[1] ]
                                
        ## Plots
        plt.figure(figsize=[17,15])
        plt.subplot(3,2,1)

        ran_lower_bound, ran_upper_bound, random_mean = self.convert_data(random)
        pid_lower_bound, pid_upper_bound, pid_mean = self.convert_data(pid)
        #episodes = np.arange(0,self.num_episodes,1)
        episodes = np.arange(0,ran_lower_bound.shape[0],1)
        
        plt.plot(random_mean,color="royalblue",label='random' )
        plt.fill_between(episodes,
                        ran_lower_bound,
                        ran_upper_bound,
                        facecolor="royalblue", 
                        alpha=0.15,
                        #label='std'
                        )
        
        plt.plot(pid_mean,color="teal", label='pid' )
        plt.fill_between(episodes,
                        pid_lower_bound,
                        pid_upper_bound,
                        facecolor="teal", 
                        alpha=0.15,
                        #label='std'
                            )
        
        plt.legend()
        plt.ylabel('running_score')
        plt.grid()
        plt.xlabel('episodes')
        #listOf_Yticks = np.arange(-700.0, 0.0, 50)
        #plt.yticks(listOf_Yticks)
        #plt.ylim(-600.0,0.0)

        # save the figure
        if self.agent_type == 1: #trained
            plt.savefig('eval_trained_plots.png', dpi=300, bbox_inches='tight')
        elif self.agent_type == 0: #random
            plt.savefig('eval_random_plots.png', dpi=300, bbox_inches='tight')
        else: #scripted
            plt.savefig('eval_script_plots.png', dpi=300, bbox_inches='tight')
        plt.show()

    def keep_time(self):
        if self.duration > 3600: 
            #print('{:.2f} hrs'.format(duration/3600))
            line = 'Script run for {:.2f} hrs'.format(self.duration/3600)
        else:
            #print('{:.2f} min'.format(duration/60))
            line = 'Script run for {:.2f} min'.format(self.duration/60)
        print(line)
    """

if __name__ == '__main__':
    returns_data = []
    data_name = ['random','pcon']
    data_dic = {
        'random': None,
        'pcon': None,
        }
    
    """
    for i in range(2):
        #print("i: ", i)
        agent = AGENT(1,0) # head=[0:GUI, 1:DIRECT] | agent_type=[0:random, 1:PID]
        returns_data.append(agent.evaluate())
        #print(returns_data)
        data_dic[ data_name[i] ] = returns_data[0]
        #print(data_dic)
        returns_data = [] #empty list for new storage
    agent.plot_multiple(data_dic)
    """

    agent = AGENT(1,0) # head=[0:GUI, 1:DIRECT] | agent_type=[0:random, 1:P-controller]
    random_returns = agent.evaluate()
    agent.plotting(random_returns)


    
