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

#from task import SingleLink as one_link_arm
from task_w_testing_data import SingleLink as one_link_uniform_data
from testing_data import data_base

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

## Model
class Critic(nn.Module): #critic_model
    def __init__(self,num_states,num_actions,num_hidden_l1,num_hidden_l2):
        super(Critic, self).__init__()
        self.ss = nn.Linear(num_states+num_actions,1)
             
    def forward(self, state, action):
        x = torch.cat([state,action],dim=-1)
        return self.ss(x)

class Actor(nn.Module): #actor_model
    def __init__(self,num_states,num_actions,num_hidden_l1,num_hidden_l2,act_limit):
        super(Actor, self).__init__()
        
        self.mean = nn.Linear(num_states,num_actions)
        self.log_std = nn.Linear(num_states,num_actions)
        self.act_limit = act_limit
          
    def forward(self, state):
        #x = self.net(state)
        x = state
        mu = self.mean(x)

        #actions
        action = torch.tanh(mu) #bound action [-1,1]
        action = self.act_limit*action #scale actions
        #print('action: ', action)

        return action #, log_pi

class AGENT():
    def __init__(self,head=0,agent_type=1):
        

        #Input1: {p.GUI=False/0, p.DIRECT=True/1} 
        #Input2:{sparse_rewards=False/0, dense_rewards=True/1}
        #self.env = one_link_arm(head) #see above comment ^
        self.env = one_link_uniform_data(head)
        self.data_issuer = data_base() #datapoints issuers
        self.agent_type = agent_type
        self.cum_rewards = []
        self.cum_running_score = []
        self.std_running_score = []
        self.num_episodes = 500 #total number of episodes
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
    
    #state normalization
    def normalization(self,obs):
        high = np.pi
        low = -np.pi
        obs = 2*((obs - low)/(high - low)) - 1
        return obs

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
        if self.agent_type == 1: #agent_type = 1, trained_agent
            #state = torch.tensor(self.state_input(state))
            action = self.actor(state)
            action = action.detach().cpu().numpy()[0]
            return action
        elif self.agent_type == 0: #agent_type = 0, random_agent
            action = self.env.action_space.sample()[0]
            return action
        else:
            kp = 1.0 #-50      
            #print('state: ', state[0].numpy()*(180/np.pi) )
            #print('goal: ', state[1].numpy()*(180/np.pi) )

            ## 180 <= angles <= 180
            #error = state[3] - state [0]
            #action = np.tanh(kp*error)

            #error = self.env.distance(state)
            #print('error: ', error*(180/np.pi))

            #""" ##0 <= angles <= 360
            error = self.env.angle_error(state)
            flag = self.error_direction(state)
            if flag: action = np.tanh(-kp*error) #CW
            else: action = np.tanh(kp*error) #CCW
            #"""
            
            #print('action: ', action.numpy())

            return action

    def state_input(self,state):
        state = state.numpy()
        return [state[0], state[3]]

    def evaluate(self):
        start_time = time.time()
        
        ## Configurations
        #hyperameters
        num_episodes = self.num_episodes
        steps = 0
        print_every = 10
        data_idx = 0

        #model
        #num_states = 2 #state_input_reduction 
        num_states = self.env.observation_space.shape[0]
        num_actions = self.env.action_space.shape[0]
        num_hidden_l1 = 2 #3 #256 
        num_hidden_l2 = 3 # 256 
        act_limit = self.env.action_space.high[0]
        #print('act_limit: ',act_limit)

        #declare model
        self.actor = Actor(num_states,
                      num_actions,
                      num_hidden_l1,
                      num_hidden_l2,
                      act_limit)


        self.critic = Critic(num_states,
                        num_actions,
                        num_hidden_l1,
                        num_hidden_l2)

        #load pre-saved model
        self.actor.load_state_dict(torch.load('sac_actor_model.pth'))
        self.critic.load_state_dict(torch.load('sac_critic_model.pth'))

        for episode in range(num_episodes):
            self.actor.eval()
            self.critic.eval()

            done = False
            steps = 0 
            score = 0

            ##issue goal & start positions
            goal_startPos = self.data_issuer.issue(data_idx)
            self.env.datapoints(goal_startPos)
            obs = self.env.reset()
            #print(obs)
            #obs = self.state_input(obs) #state_input_reduction
            data_idx += 1
            #print('points: ',goal_startPos*(180/np.pi))

            #if self.agent_type == 1: #agent_type = 1, trained_agent, normalize
            #    obs = self.normalization(obs)
            #print('obs: ', obs)

            state = torch.tensor(obs).float() 
            #print('state: ', state)
            
            while not done:
                steps += 1
                action = self.output_action(state) #trained_agent
                #print('action: ', action)
                next_obs, reward, done, info = self.env.step(action)
                score += reward
                #print('distance: ',info['distance'])
                #print('state: ', state)
                #print(next_obs)

                #if self.agent_type == 1: #agent_type = 1, trained_agent
                #    next_obs = self.normalization(next_obs)

                next_state = torch.tensor(next_obs).float() 
                state = next_state

                time.sleep(.01)

            self.cum_rewards.append(score)
            #avg_reward = np.mean(self.cum_rewards[-self.window_size:])
            #std_reward = np.std(self.cum_rewards[-self.window_size:])
            #self.cum_running_score.append(avg_reward)
            #self.std_running_score.append(std_reward)
            
            if episode % print_every == 0:
                #print("Episode: {} | Avg_reward: {}".format(episode,score))
                print("Episode: {} | Avg_reward: {} | steps: {}".format(episode,score,info['step']))
        
        #print(self.cum_running_score)
        end_time = time.time()
        self.duration = end_time - start_time
        #print('evaluation_done')
        return self.cum_rewards #self.cum_running_score, self.std_running_score

    def convert_data(self,cum_rewards):
        value = np.lib.stride_tricks.sliding_window_view(cum_rewards,self.window_size)
        cum_running_score = np.mean(value,axis=-1)
        std_running_score = np.std(value,axis=-1)
        
        lower_bound = np.array(cum_running_score) - np.array(std_running_score)
        upper_bound = np.array(cum_running_score) + np.array(std_running_score) 
        return lower_bound, upper_bound, cum_running_score       

    def plotting(self,data_dic):

        random, agent, pid = None, None, None
        data_name = ['random','agent','pid']

        if data_dic[ data_name[0] ] is not None:
            random  = data_dic[ data_name[0] ]

        if data_dic[ data_name[1] ] is not None:
            agent  = data_dic[ data_name[1] ]

        if data_dic[ data_name[2] ] is not None:
            pid  = data_dic[ data_name[2] ]
                                
        ## Plots
        plt.figure(figsize=[17,15])
        plt.subplot(3,2,1)

        if random is None:
            age_lower_bound, age_upper_bound, agent_mean = self.convert_data(agent)
            pid_lower_bound, pid_upper_bound, pid_mean = self.convert_data(pid)
            #episodes = np.arange(0,self.num_episodes,1)
            episodes = np.arange(0,age_lower_bound.shape[0],1)

            plt.plot(agent_mean,color="tomato",label='agent'  )
            plt.fill_between(episodes,
                            age_lower_bound,
                            age_upper_bound,
                            facecolor="tomato", 
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
        else:
            ran_lower_bound, ran_upper_bound, random_mean = self.convert_data(random)
            age_lower_bound, age_upper_bound, agent_mean = self.convert_data(agent)
            pid_lower_bound, pid_upper_bound, pid_mean = self.convert_data(pid)
            #episodes = np.arange(0,self.num_episodes,1)
            episodes = np.arange(0,age_lower_bound.shape[0],1)
            
            plt.plot(random_mean,color="royalblue",label='random' )
            plt.fill_between(episodes,
                            ran_lower_bound,
                            ran_upper_bound,
                            facecolor="royalblue", 
                            alpha=0.15,
                            #label='std'
                            )

            plt.plot(agent_mean,color="tomato",label='agent'  )
            plt.fill_between(episodes,
                            age_lower_bound,
                            age_upper_bound,
                            facecolor="tomato", 
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

if __name__ == '__main__':
    data = []
    data_name = ['random','agent','pid']
    data_dic = {
        'random': None,
        'agent': None,
        'pid': None,
        }

    for i in range(1):
    #for i in range(3):
    #for i in range(1,3):
        agent = AGENT(0,i) # head=[0:GUI, 1:DIRECT] | agent_type=[0:random, 1:trained, 2:PID]
        data.append(agent.evaluate())
        data_dic[ data_name[i] ] = data[0]
        data = [] #empty list for new storage
    agent.plotting(data_dic)
    
