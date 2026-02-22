import random
from copy import copy, deepcopy
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

sys.path.insert(0, "..")
from envs.task import SingleLink as one_link_arm

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


## Memory Replay
Transition = namedtuple('Transition', ('state','next_state','action','reward', 'done'))
                        
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

        self.net = nn.Sequential(
            nn.Linear(num_states+num_actions,num_hidden_l1),
            nn.ReLU(),
            nn.Linear(num_hidden_l1,num_hidden_l2),
            nn.ReLU(),
            nn.Linear(num_hidden_l2,1),          
            )
        
        #initialization
        for i,m in enumerate( self.modules() ):
            if isinstance(m, nn.Linear):
                if i == 6:
                    nn.init.uniform_(m.weight, -3e-3, 3e-3 )
                else:
                    nn.init.uniform_(m.weight, 
                                    *self.hidden_init(m) )
          
    def hidden_init(self, layer):
        fan_in = layer.weight.data.size()[0]
        lim = 1.0/np.sqrt(fan_in)
        return (-lim,lim)   
        
    def forward(self, state, action):
        x = self.net( torch.cat([state,action],1) )
        return x

class Actor(nn.Module): #actor_model
    def __init__(self,num_states,num_actions,num_hidden_l1,num_hidden_l2):
        super(Actor, self).__init__()
        
        self.net = nn.Sequential(
            nn.Linear(num_states,num_hidden_l1),
            nn.ReLU(),
            nn.Linear(num_hidden_l1,num_hidden_l2),
            nn.ReLU(),
            nn.Linear(num_hidden_l2,num_actions),
            nn.Tanh(),           
            )

        #initialization
        for i,m in enumerate( self.modules() ):
            if isinstance(m, nn.Linear):
                if i == 6:
                    nn.init.uniform_(m.weight, -3e-3, 3e-3 )
                else:
                    nn.init.uniform_(m.weight, 
                                    *self.hidden_init(m) )
        
    def hidden_init(self, layer):
        fan_in = layer.weight.data.size()[0]
        lim = 1.0/np.sqrt(fan_in)
        return (-lim,lim) 

    def forward(self, state):
        x = self.net(state)
        return x

## OUActionNoise
class OUActionNoise:
    def __init__(self, mean=0, std=0.3, theta=.15, dt=1e-2, x0=None):
        self.theta = theta
        self.mean = mean
        self.std = 1.0*std
        self.dt = dt
        self.x0 = x0
        self.reset()
        
    def __call__(self):
        x = self.x_prev + ( 
            self.theta*(self.mean - self.x_prev)*self.dt + 
            self.std*np.sqrt(self.dt)*np.random.normal(size=self.mean.shape)
            )
        self.x_prev = x
        return x
                           
    def reset(self):
        if self.x0 is not None:
            self.x_prev = self.x0
        else:
            self.x_prev = np.zeros_like(self.mean)

class DDPG_Solver():
    def __init__(self, head=0, n_eps=750):
        
        #Input1: {p.GUI=False/0, p.DIRECT=True/1} 
        #Input2:{sparse_rewards=False/0, dense_rewards=True/1}
        self.env = one_link_arm() #see above comment ^
        
        self.cum_rewards = []
        self.cum_running_score = []
        self.std_running_score = []
        self.num_episodes = n_eps #total number of episodes
        self.window_size = 100 #rolling window size
        self.epsilon_cum = []
        self.avg_act_loss = []
        self.avg_cri_loss = []
        self.actor_losses = []
        self.critic_losses = []
        self.duration = None 
        self.success_array = []
    
    ### Re-setting all parameters
    def seed_everything(self, seed: int):
        import random
        import numpy as np
        import torch
        
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)

    ## Policy 
    def output_action(self, actor, state, eps, mean):
        ou_noise = OUActionNoise(mean)
        action = actor(state).detach().cpu().numpy()[0]
        action += ou_noise()*max(0,eps)
        action = np.clip(action,-1.,1.)
        return action

    ## Immediate update
    def update_target_model(self, net, target_net):
        target_net.load_state_dict(net.state_dict())
        target_net.eval()

    ## Polyak update
    def smooth_update_target_model(self, net, target_net, tau):
        #rho = 1 - (1/tau) #decay rate 
        for p_tgt, p in zip(target_net.parameters(),net.parameters()):
            #p_tgt.data.mul_(1.0 - rho)
            #p_tgt.data.add_(rho*p.data) 
            p_tgt.data.copy_(tau * p.data +  (1.0 - tau) * p_tgt.data)
            

    ## Train model
    def train_model(self, 
                    batch, 
                    q_optimizer,
                    policy_optimizer,
                    MSE,
                    batch_size,
                    tgt_actor,
                    actor,
                    tgt_critic,
                    critic,
                    gamma,
                    ):
    
        states = torch.stack(batch.state) #torch.cat(batch.state) 
        next_states = torch.stack(batch.next_state) #torch.cat(batch.next_state)
        actions = torch.cat(batch.action) #actions = torch.tensor(batch.action).reshape(batch_size,-1)
        rewards = torch.tensor(batch.reward).reshape(batch_size,-1)
        rewards = rewards.float()
        dones = torch.tensor(batch.done).reshape(batch_size,-1)
        dones = dones.float()

        #print('states: ', states)
        #print('dones: ', dones)
        #print('rewards: ',rewards)

        next_actions = tgt_actor(next_states)

        #print('next_actions: ',next_actions)

        
        #Q = Q(s,a|Q-net)
        qvalues = critic(states,actions) #.squeeze(1)
        #print('qvalues: ',qvalues)

        
        #Q' = Q(s',a'|Q-net)
        next_qvalues = tgt_critic(next_states, next_actions) #s_t+1, #u(s_t+1) #.squeeze(1)
        #print('next_qvalues: ', next_qvalues)

        # y = r + Y*Q'
        target = rewards + (1.0 - dones)*gamma*next_qvalues.detach()
        #print('target: ' ,target)

        #puase
        ##critic loss
        # L = (1/N)*sum(y - Q)
        loss_critic = MSE(qvalues, target)
        #print('loss_critic: ',loss_critic)

        #minimize the loss
        q_optimizer.zero_grad()
        loss_critic.backward() 
        q_optimizer.step()

        ##policy loss
        # J = (1/N)*sum( Q(s,u| Q-net) ) where u = u(s|P-net)
        loss_actor = -critic(states,actor(states))
        loss_actor = loss_actor.mean()

        #maximize the loss
        policy_optimizer.zero_grad()
        loss_actor.backward()
        policy_optimizer.step() 
        
        return loss_actor.item(), loss_critic.item()

    def main(self):
        start_time = time.time()
        
        ## Configurations
        #hyperameters
        batch_size = 128 #64 
        tau = 1e-3 #[0,1]
        replay_memory_capacity = int(1e6)
        memory = Memory(replay_memory_capacity)
        initial_exploration = 128
        gamma = 0.99 #0.995
        lr_actor = 1e-3
        lr_critic = 1e-3
        epsilon = 1.0
        num_episodes = self.num_episodes
        steps = 0
        print_every = 10

        #model 
        num_states = self.env.observation_space.shape[0]
        num_actions = self.env.action_space.shape[0]
        num_hidden_l1 = 256 #250
        num_hidden_l2 = 256 #150 

        mean = np.zeros(num_actions) #hyperparameter

        #declare model
        actor = Actor(num_states,
                      num_actions,
                      num_hidden_l1,
                      num_hidden_l2,
                      )

        critic = Critic(num_states,
                        num_actions,
                        num_hidden_l1,
                        num_hidden_l2,
                        )

        tgt_actor = Actor(num_states,
                          num_actions,
                          num_hidden_l1,
                          num_hidden_l2,
                          )

        tgt_critic = Critic(num_states,
                            num_actions,
                            num_hidden_l1,
                            num_hidden_l2,
                            )

        #initialize target networks
        tgt_actor = deepcopy(actor) #copy weights
        tgt_critic = deepcopy(critic) #copy weights

        #optimizer & loss function
        q_optimizer = optima.Adam(critic.parameters(), lr=lr_critic)
        policy_optimizer = optima.Adam(actor.parameters(), lr=lr_actor)
        MSE = nn.MSELoss()

        for episode in range(num_episodes):
        #for episode in range(10):
            done = False
            steps = 0 
            score = 0

            obs = self.env.reset()
            #print('obs: ', obs)
            state = torch.tensor(obs).float() 
            #print('state: ', state)
            
            while not done:
            #for _ in range(2):
                steps += 1
                #action = self.env.action_space.sample()
                action = self.output_action(actor,state,epsilon,mean)
                #print('action: ', action)
                next_obs, reward, done, info = self.env.step(action)
                next_state = torch.tensor(next_obs).float() 
                #print('next_state: ',next_state)
                #print('raw_rewards: ',reward)
                #print('done: ', done)
                
                mask = 0 
                score += reward #reward

                if done:
                    mask = 1

                action = torch.tensor(action).unsqueeze(0).float() #convert list to tensor
                memory.push(state, next_state, action, reward/25, mask)
                state = next_state           
            #"""
                if steps > initial_exploration:
                    #epsilon = max(epsilon*0.9995, 0.01)
                    batch = memory.sample(batch_size)
                    pol_loss, val_loss = self.train_model(batch, 
                                                    q_optimizer,
                                                    policy_optimizer,
                                                    MSE,
                                                    batch_size,
                                                    tgt_actor,
                                                    actor,
                                                    tgt_critic,
                                                    critic,
                                                    gamma,
                                                    )
                    self.actor_losses.append(pol_loss)
                    self.critic_losses.append(val_loss)
                
                #smooth_tgt_update
                self.smooth_update_target_model(tgt_actor,actor,tau) 
                self.smooth_update_target_model(tgt_critic,critic,tau)
            #print('steps: {}, rewards: {}'.format(steps,score))

            if self.actor_losses: #execute if not empty
                self.avg_act_loss.append( np.mean(self.actor_losses) )
                self.avg_cri_loss.append( np.mean(self.critic_losses) )
            
            self.epsilon_cum.append(epsilon)
            
            score = score #if score == 500.0 else score #+ 1
            #print('score: ', score)
            self.cum_rewards.append(score)
            avg_reward = np.mean(self.cum_rewards[-self.window_size:])
            std_reward = np.std(self.cum_rewards[-self.window_size:])
            self.cum_running_score.append(avg_reward)
            self.std_running_score.append(std_reward)
            
            if episode % print_every == 0:
                print("Episode: {} | Avg_reward: {}".format(episode,avg_reward))
            #"""
   
        end_time = time.time()
        self.duration = end_time - start_time
        
        #prep_save model
        self.save_actor = deepcopy(actor) #copy weights
        self.save_critic = deepcopy(critic) #copy weights

        #self.env.close() #close simulator

    def plotting(self):
        ## Plots
        """
        plt.figure(figsize=[17,15])
        plt.subplot(3,2,1)
        plt.plot(self.cum_rewards)
        plt.ylabel('rewards')
        #plt.xlabel('episodes')
        """
        plt.figure(figsize=[17,15])
        colour = "tomato"
        lower_bound = np.array(self.cum_running_score) - np.array(self.std_running_score)
        upper_bound = np.array(self.cum_running_score) + np.array(self.std_running_score)
        episodes = np.arange(0,self.num_episodes,1)
        plt.subplot(3,2,1)
        plt.plot(self.cum_running_score, label="mean", color=colour )
        plt.fill_between(episodes,lower_bound,upper_bound,facecolor=colour, alpha=0.15,label='std')
        plt.ylabel('running_score')
        plt.grid()
        plt.xlabel('episodes')
        #plt.legend()
        listOf_Yticks = np.arange(-200.0, 10.0, 15)
        plt.yticks(listOf_Yticks)
        plt.ylim(-200.0,10.0)
        """
        plt.subplot(3,2,3)
        plt.plot(self.epsilon_cum) 
        plt.ylabel('epsilon')
        plt.xlabel('episodes')

        plt.subplot(3,2,4)
        plt.plot(self.avg_loss) 
        plt.ylabel('avg_losses')
        plt.xlabel('episodes')
        """
        # save the figure
        plt.savefig('plots.png', dpi=300, bbox_inches='tight')
        plt.show()

    def keep_time(self):
        if self.duration > 3600: 
            #print('{:.2f} hrs'.format(duration/3600))
            line = 'Script run for {:.2f} hrs'.format(self.duration/3600)
        else:
            #print('{:.2f} min'.format(duration/60))
            line = 'Script run for {:.2f} min'.format(self.duration/60)

        #write to file
        outF = open('details.txt', 'w')
        outF.write(line)
        outF.close()
        print(line)

    def save_model(self):
        torch.save(self.save_actor.state_dict(), 'saved_models/actor_model.pth')
        torch.save(self.save_critic.state_dict(), 'saved_models/critic_model.pth')

if __name__ == '__main__':
    agent = DDPG_Solver(1,1000) #head=[0:GUI,1:DIRECT] | num_episodes
    #agent.seed_everything(42)
    agent.main()
    agent.save_model()
    agent.plotting()
    agent.keep_time()
