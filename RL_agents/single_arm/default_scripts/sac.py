from operator import inv
import random
from copy import deepcopy
import numpy as np
from collections import namedtuple, deque
import matplotlib.pyplot as plt
import time
import sys
#pytorch
import torch
import torch.nn as nn
import torch.optim as optima
import torch.nn.functional as F
from torch.distributions.normal import Normal

sys.path.insert(0, "..")
from envs.task import SingleLink as one_link_arm

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


## Memory Replay
Transition = namedtuple('Transition', ('state','next_state','action','reward','done'))
                        
class Memory(object):
    def __init__(self, capacity):
        self.memory = deque(maxlen=capacity)
        self.capacity = capacity

    def push(self, state, next_state, action, reward, done):
        self.memory.append(Transition(
            state, next_state, action, reward, done))    
    def sample(self, batch_size):

        transitions = random.sample(self.memory, batch_size)
        batch = Transition(*zip(*transitions))
        return batch
    
    def __len__(self):
        return len(self.memory)

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
        x = state
        mu = self.mean(x)
        log_std = self.log_std(x)

        log_std = torch.clamp(log_std,1e-6,1) #avoid -ve exponentials
        #log_std = torch.clamp(log_std,-20,2)
        std  = torch.exp(log_std)
        #print('std: ', std)

        pi_distribution = Normal(mu,std)
        action_u = pi_distribution.rsample() #sample actions

        #log_likelihood
        log_mu = pi_distribution.log_prob(action_u)
        log_pi = log_mu - (2*(np.log(2) - action_u - F.softplus(-2*action_u))) 

        #actions
        action = torch.tanh(action_u) #bound action [-1,1]
        action = self.act_limit*action #scale actions
        #print('action: ', action)

        return action, log_pi

class SAC_Solver():
    def __init__(self, head=0, n_eps=500):
        
        #Input1: {p.GUI=False/0, p.DIRECT=True/1} 
        #Input2:{sparse_rewards=False/0, dense_rewards=True/1}
        self.env = one_link_arm(head) #see above comment ^
        
        self.cum_rewards = []
        self.cum_running_score = []
        self.std_running_score = []
        self.num_episodes = 500 #total number of episodes
        self.window_size = 50 #rolling window size
        self.epsilon_cum = []
        self.avg_pi_loss = []
        self.avg_q1_loss = []
        self.avg_q2_loss = []
        self.pi_losses = []
        self.q1_losses = []
        self.q2_losses = []
        self.duration = None 
        self.batch_size = 64
        self.gamma = 0.995
        self.alpha = 0.1

    def initialize_weights(self,m):
        if isinstance(m,nn.Linear):
            nn.init.uniform_(m.weight.data, -3e-4, 3e-4 )
            nn.init.constant_(m.bias.data,0)

    ### Re-setting all parameters
    def seed_everything(self, seed: int):
        import random
        import numpy as np
        import torch
        
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
    
    #state normalization
    def normalization(self,obs):
        high = np.pi
        low = -np.pi
        obs = 2*((obs - low)/(high - low)) - 1
        return obs

    ## Policy 
    def get_action(self, state):
        #state = torch.tensor(self.state_input(state))
        with torch.no_grad():
            action, _ = self.pi_net(state)
            #print('action:' ,action)
            return action[0].numpy()

    ## Immediate update
    def update_target_model(self, net, target_net):
        target_net.load_state_dict(net.state_dict())
        target_net.eval()

    ## Polyak update
    def smooth_update_target_model(self, net, target_net):
        rho = 1 - (1/self.tau) #decay rate 
        for p_tgt, p in zip(target_net.parameters(),net.parameters()):
            p_tgt.data.mul_(1.0 - rho)
            p_tgt.data.add_(rho*p.data) 

    #compute Q losses
    def compute_loss_q(self,states,actions,next_states,rewards,dones):
        #print('q_states: ',states)
        #print('q_actions: ', actions)
        # print('next_states: ',next_states)
        combined = torch.cat([states,actions], dim=-1)
        #print('q_states-action: ', combined)
        # test = self.pi_net(next_states)
        # print('test passed')

        q1 = self.q1_net(states,actions)
        q2 = self.q2_net(states,actions)
        #print('q1: ',q1)

        with torch.no_grad():
            next_actions, next_log_pis = self.pi_net(next_states) #sample actions

            tgt_q1 = self.tgt_q1_net(next_states,next_actions)
            tgt_q2 = self.tgt_q2_net(next_states,next_actions)
            tgt_q = torch.min(tgt_q1,tgt_q2)

            #print('tgt_q: ', tgt_q)
            #print('next_log_pis: ', next_log_pis)
            #print('rewards: ', rewards)
            #diff = self.gamma*(tgt_q - next_log_pis)
            #print('diff: ',  diff)
            #print('r + diff: ', rewards+diff )

            y = rewards + self.gamma*(1.0 - dones)*(tgt_q - self.alpha*next_log_pis)
            #y = y.squeeze(0)
            #print('y: ', y)

        loss_q1 = ((q1-y)**2).mean() 
        loss_q2 = ((q2-y)**2).mean() 
        #print('loss_q1: ', loss_q1)

        return loss_q1, loss_q2

    #compute Policy losses
    def compute_loss_pi(self,states):
        actions,log_pis = self.pi_net(states) #sample actions
        q1 = self.q1_net(states,actions)
        q2 = self.q2_net(states,actions)
        q = torch.min(q1,q2)

        mod_log_pis = self.alpha*log_pis.squeeze(0)

        #print('q_a: ', q)
        #print('a*log_pi: ', mod_log_pis)
        
        loss_pi = (q - mod_log_pis).mean()
        #print('loss_pi: ',loss_pi)

        return -loss_pi #-ve for gradient ascent

    ## Train model
    def train_model(self, 
                    batch, 
                    pi_optim,
                    q1_optim,
                    q2_optim):
    
        states =  torch.stack(batch.state) #torch.cat(batch.state) #
        next_states = torch.stack(batch.next_state) #torch.cat(batch.next_state)
        #actions = torch.cat(batch.action) 
        actions = torch.tensor(batch.action).reshape(self.batch_size,-1)
        rewards = torch.tensor(batch.reward).reshape(self.batch_size,-1)
        rewards = rewards.float()
        dones = torch.tensor(batch.done).reshape(self.batch_size,-1)
        dones = dones.float()

        #print('next_states: ', next_states)
        #print('states: ', states)
        #print('actions: ', actions)
        #print('rewards: ', rewards)
        #print('dones: ', dones)

        #Q-functions updates
        q1_optim.zero_grad() #clear data
        q2_optim.zero_grad() #clear data

        loss_q1, loss_q2 = self.compute_loss_q(states,actions,next_states,rewards,dones)
        loss_q1.backward() #gradients
        loss_q2.backward() #gradients
        q1_optim.step() #update
        q2_optim.step() #update

        #for param in self.q1_net.parameters(): print('Q1-network: ',param.data)
        #for param in self.q2_net.parameters(): print('Q2-network: ',param.data)
 
        #freeze Q-networks to avoid computing during policy update
        for p in self.q1_net.parameters():
            p.requires_grad = False
        for p in self.q2_net.parameters():
            p.requires_grad = False 

        #policy updates
        pi_optim.zero_grad() #clear data
        loss_pi = self.compute_loss_pi(states)
        #print('loss_pi: ', loss_pi)
        loss_pi.backward() #gradients
        pi_optim.step() #update

        #for param in self.pi_net.parameters(): print('policy: ',param.data)

        #unfreeze Q-networks to avoid computing on next iteration
        for p in self.q1_net.parameters():
            p.requires_grad = True
        for p in self.q2_net.parameters():
            p.requires_grad = True

        #polyak updating of target networks
        with torch.no_grad():
            self.smooth_update_target_model(self.q1_net, self.tgt_q1_net)
            self.smooth_update_target_model(self.q2_net, self.tgt_q2_net)
        
        return loss_pi.item(), loss_q1.item(), loss_q2.item()

    def state_input(self,state):
        state = state.numpy()
        return [state[0], state[3]]

    def main(self):
        start_time = time.time()
        
        #hyperameters
        batch_size = self.batch_size
        self.tau = 200
        replay_memory_capacity = int(4e3)
        memory = Memory(replay_memory_capacity)
        initial_exploration = int(15*500) #eps*total_eps_steps 
        lr = 1e-1 #3e-4 
        num_episodes = self.num_episodes
        steps = 0
        print_every = 10
        
        #model 
        #num_states = 2 #state_input_reduction 
        num_states = self.env.observation_space.shape[0]
        num_actions = self.env.action_space.shape[0]
        num_hidden_l1 = 2 #3 #256 
        num_hidden_l2 = 3 # 256 
        act_limit = self.env.action_space.high[0]
        #print('num_states: ',num_states)
        #print('num_actions: ',num_actions)
        #print('act_limit: ',act_limit)

        #declare model
        self.pi_net = Actor(num_states,
                      num_actions,
                      num_hidden_l1,
                      num_hidden_l2,
                      act_limit)

        self.q1_net = Critic(num_states,
                        num_actions,
                        num_hidden_l1,
                        num_hidden_l2)

        self.q2_net = Critic(num_states,
                       num_actions,
                       num_hidden_l1,
                       num_hidden_l2)

        #initialize weights
        self.pi_net.apply(self.initialize_weights)
        self.q1_net.apply(self.initialize_weights)
        self.q2_net.apply(self.initialize_weights)

        #target networks
        self.tgt_q1_net = deepcopy(self.q1_net)
        self.tgt_q2_net = deepcopy(self.q1_net)

        #freeze target networks to only update via polyak averaging
        for p in self.tgt_q1_net.parameters():
            p.requires_grad = False
        for p in self.tgt_q2_net.parameters():
            p.requires_grad = False           

        #optimizer & loss function
        pi_optim = optima.Adam(self.pi_net.parameters(), lr=lr)
        q1_optim = optima.Adam(self.q1_net.parameters(), lr=lr)
        q2_optim = optima.Adam(self.q2_net.parameters(), lr=lr)

        for episode in range(num_episodes):
            done = False
            #steps = 0 
            score = 0

            #obs = self.env.reset()
            obs = self.env.reset()
            #print('obs_before: ', obs)
            #obs = self.state_input(obs)
            #print('obs: ', obs*(180/np.pi) )
            state = torch.tensor(obs).float() 
            #print('state: ', state)
            
            while not done:
                steps += 1
                #action = self.get_action(state)
                action = self.env.action_space.sample()[0]
          
                #print('action: ', action)

                next_obs, reward, done, info = self.env.step(action)
                #next_obs = self.state_input(next_obs)
                next_state = torch.tensor(next_obs).float()  
                #print('next_state: ',next_state)
                #print('raw_rewards: ',reward)
                #print('steps: ', steps)
                #print('done: ', done)
                #print('distance: ', info['distance'])
                
                mask = 0 
                score += reward
                reward = reward/self.alpha #scaling for exploration vs. exploitation
                #print('scaled_reward: ', reward)

                if done:
                    mask = 1

                action = torch.tensor(action).unsqueeze(0).float() #convert list to tensor
                memory.push(state, next_state, action, reward, mask)
                
                """###Next method
                next_state_memory = torch.tensor(self.state_input(next_state))
                state_memory = torch.tensor(self.state_input(state))
                memory.push(state_memory, next_state_memory, action, reward, mask)
                """
                state = next_state   

                #time.sleep(.01) 
                #print('steps: ', steps)
                if steps > initial_exploration:
                    #print('steps: ', steps)

                    if steps == initial_exploration + 10 :
                        pass
                        #print('intial_pass, episode: ', episode) 
                        #return

                    batch = memory.sample(batch_size)
                    pi_loss, q1_loss, q2_loss = self.train_model(batch, 
                                                                 pi_optim,
                                                                 q1_optim,
                                                                 q2_optim,)
                    self.pi_losses.append(pi_loss)
                    self.q1_losses.append(q1_loss)
                    self.q2_losses.append(q2_loss)
                    
            if self.pi_losses: #execute if not empty
                self.avg_pi_loss.append( np.mean(self.pi_losses) )
                self.avg_q1_loss.append( np.mean(self.q1_losses) )
                self.avg_q2_loss.append( np.mean(self.q2_losses) )

            self.cum_rewards.append(score)
            if episode % print_every == 0:
                #print("Episode: {} | Avg_reward: {}".format(episode,score))
                print("Episode: {} | Avg_reward: {} | steps: {}".format(episode,score,info['step']))

        end_time = time.time()
        self.duration = end_time - start_time

        #prep_save model
        self.save_actor = deepcopy(self.pi_net) #copy weights
        self.save_critic = deepcopy(self.q1_net) #copy weights

    #rolling window
    def roll_avg_std(self,array,size):
        value = np.lib.stride_tricks.sliding_window_view(array,size)
        avg = np.mean(value,axis=-1)
        std = np.std(value,axis=-1)
        return avg, std

    ## Plots
    def plotting(self):
        roll_avg_std = self.roll_avg_std(self.cum_rewards,self.window_size)
        self.cum_running_score,self.std_running_score = roll_avg_std
        episodes = np.arange(0,self.std_running_score.shape[0],1)

        plt.figure(figsize=[17,15])
        colour = "tomato"
        lower_bound = np.array(self.cum_running_score) - np.array(self.std_running_score)
        upper_bound = np.array(self.cum_running_score) + np.array(self.std_running_score)
        
        plt.subplot(3,2,1)
        plt.plot(self.cum_running_score, label="mean", color=colour )
        plt.fill_between(episodes,lower_bound,upper_bound,facecolor=colour, alpha=0.15,label='std')
        plt.ylabel('running_score')
        plt.grid()
        plt.xlabel('episodes')
        listOf_Yticks = np.arange(-700.0, 10.0, 100)
        plt.yticks(listOf_Yticks)
        plt.ylim(-700.0,10.0)

        # save the figure
        plt.savefig('plots.png', dpi=300, bbox_inches='tight')
        plt.show()

    def keep_time(self):
        if self.duration > 3600: 
            line = 'Script run for {:.2f} hrs'.format(self.duration/3600)
        else:
            line = 'Script run for {:.2f} min'.format(self.duration/60)

        #write to file
        outF = open('details.txt', 'w')
        outF.write(line)
        outF.close()
        print(line)

    def save_model(self):
        torch.save(self.save_actor.state_dict(), 'sac_actor_model.pth')
        torch.save(self.save_critic.state_dict(), 'sac_critic_model.pth')

#Execution
if __name__ == '__main__':
    agent = SAC_Solver(1,500) #head=[0:GUI,1:DIRECT] | num_episodes
    agent.main()
    agent.save_model()
    agent.plotting()
    agent.keep_time()
