import pybullet as p
import pybullet_data
import gym
from gym import spaces
# import gymnasium as gym
# from gymnasium import spaces
import numpy as np
from numpy import random
import time
from os.path import dirname, abspath, join

##urdf location
this_dir = dirname(__file__)
# arm_dir = abspath(join(this_dir,'arm2.urdf')) #L1:1.65
arm_dir = abspath(join(this_dir,'arm.urdf')) #L1:1.00

class SingleLink(gym.Env):

    ##initiation
    def __init__(self,head=True):
        # print("Correct MOO.py")
        # xx
        self.connect(head)
        high_state = np.pi 
        self.observation_space = spaces.Box(-high_state, 
                                             high_state, 
                                             shape=(9,), 
                                             dtype='float32')

        self.action_space = spaces.Box(low=-1.0, 
                                       high=1.0, 
                                       shape=(1,), 
                                       dtype='float32')
        
    def connect(self,head):
        if head:
            p.connect(p.DIRECT)
        else:
            p.connect(p.GUI)
            
    ##primary modules
    def step(self,action):
        self.action_scale = 45.00 #25.00 #35.00 #30.00 
        self.step_id += 1
        self.jointTorque = np.array(action)*self.action_scale
        self.move(self.jointTorque)
        self.simulate(1)
        state = self.compute_state()

        # print('step_state: ', state)

        # Batch(obs=i, act=i, rew=i, terminated=0, truncated=0, obs_next=i + 1, info={})
        # print(self.is_done(state))
        return state, \
               self.reward(action,state), \
               self.is_done(state), \
               0, \
               {'distance': self.distance(state),
                'step': self.step_id}
        # return state

    def reset(self):
        self._reset_world()
        self.radius = 0.96
        self.theta_pos = random.uniform(low = 0, high = 2*np.pi)
        x_pos = self.radius*np.cos(self.theta_pos)
        y_pos = self.radius*np.sin(self.theta_pos)
        objectPos = [x_pos,y_pos,0.1]
        objectOrn = p.getQuaternionFromEuler([0,0,0])
        self.objectID = p.loadURDF("sphere_small.urdf",
                    objectPos,
                    objectOrn,                    
                    useFixedBase = 1,
                    globalScaling = 0.9)
        self.simulate(50)
        state = self.compute_state()

        return state
    
    def render(self, mode='human'):
        pass
        	
    def simulate(self, num):
        for _ in range(num):
            p.stepSimulation()

    def _reset_world(self):
        p.setAdditionalSearchPath(pybullet_data.getDataPath()) #optionally 
        p.resetSimulation()
        p.setGravity(0,0,-10)
        p.setTimeStep(0.01) #former_env
        p.setRealTimeSimulation(0)
        planeId = p.loadURDF("plane.urdf")
        armStartPos = [0.0, 0.0, 0.025]
        armStartOrn = p.getQuaternionFromEuler([0.0, 0.0, 0.0])
        self.armID = p.loadURDF(arm_dir, #"arm.urdf",
                    armStartPos,
                    armStartOrn,
                    useFixedBase = 1,)
                
        joint_type_name = ['REVOLUTE', 'PRISMATIC', 'SPHERICAL', 'PLANAR', 'FIXED']
        self.joints = []
        self.links = {}
        num_joints = p.getNumJoints(self.armID)

        for joint_idx in range(num_joints):
            info = p.getJointInfo(self.armID,0)
            data = {
                'joint_idx' : info[0],
                'joint_name' : info[1].decode('utf-8'),
                'joint_type' : joint_type_name[ info[2] ],
                'link_name' : info[12].decode('utf-8'),}

            if data['joint_type'] != 'FIXED':
                self.joints.append(data)
                self.links[data['link_name']] = joint_idx
        self.step_id = 0
        self.distance_threshold = 0.05 #5cm precision /0.05

        p.setJointMotorControl2(self.armID,
                                jointIndex = 0,
                                controlMode = p.VELOCITY_CONTROL,
                                force = 0 )

        p.resetJointState(self.armID,
                          jointIndex = 0,
                          targetValue = random.uniform(0, 2*np.pi), 
                          targetVelocity = 0.0 )

    def compute_state(self):
        joint_state = p.getJointState(self.armID, jointIndex = 0)
        jointPos = self.convert(joint_state[0])
        jointVel = joint_state[1]
        s0 = jointPos #state_angle
        self.l = 0.95 #link_length
        l = self.l #link_length
        x = l*np.cos(s0) #end_effector_coord
        y = l*np.sin(s0) #end_effector_coord
        s1,s2 = x,y #end_effector_pos
        s3 = self.convert(self.theta_pos) #goal_angle
        s4 = self.radius*np.cos(s3) #goal_coord_x
        s5 = self.radius*np.sin(s3) #goal_coord_y
        s6 = x - s4
        s7 = y - s5
        s8 = jointVel #state_angular_speed
        return s0,s1,s2,s3,s4,s5,s6,s7,s8

    def convert(self, angle):        
        angle = np.round(angle,2)
        return angle % (2*np.pi)

    def angle_error(self,state):
        l = self.l #link_length
        r = self.radius #goal_radius
        d = np.sqrt( state[6]**2 + state[7]**2 )
        den = 2*l*r
        num = l**2 + r**2 - d**2
        if np.abs(num/den) > 1.0: error  = np.pi 
        else: error = np.arccos( num/den )
        return error 

    def move(self,jointInput):
        p.setJointMotorControl2(self.armID,
                        jointIndex = 0,
                        controlMode = p.TORQUE_CONTROL,
                        force = jointInput,
                        )
        self.simulate(50)
        #"""
    
    def distance(self, state, type=1):
        if type == 0: #using angles
            dis = np.linalg.norm( state[0] - state[3] )
        else: #using coordinates
            dis = np.sqrt( state[6]**2 + state[7]**2 )
        return dis

    def reward(self, jointInput, state, dense=True):
        distance  = self.distance(state)
        TorqueControl = np.square(jointInput).sum() 

        if dense:
            return -( distance**2 + TorqueControl)
            #return -(distance**2)
        else:
            return -(distance < self.distance_threshold).astype(np.float32)  
    
    def is_done(self,state):
        distance  = self.distance(state) 
        return self.step_id == 50 or distance < self.distance_threshold #former_env

#Execution
if __name__ == '__main__':
    print("single_link")
    sim = SingleLink(0)
    obs = sim.reset()

    #"""
    for _ in range(10000000000):    
        action = random.uniform(low=0.0,high=1.0)
        # action = 0
        # action = np.array(0.5)
        # print('action: ',action)
        state = sim.compute_state()
        #print('state: ', state )
        state,rew,_,_,_ = sim.step(action)

        print('sim.step(action): ', sim.step(action))

        a_error = sim.angle_error(state)
        # print('rew ', rew)
        # print('a_error: ', a_error)
        distance = sim.distance(state)
        # print('distance: ', distance)
        #print('outward: ', outward)
        #print('______________________________')
        time.sleep(0.2)
    #"""

