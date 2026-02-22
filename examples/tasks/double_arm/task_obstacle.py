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
import sys

##urdf location
this_dir = dirname(__file__)
arm_dir = abspath(join(this_dir,'arm.urdf'))
sys.path.insert(0,this_dir)
from kinematics import InverseKinematics as InvK

class DoubleLink(gym.Env):
    ##initiation
    def __init__(self,head=True):
        self.connect(head)
        high_state = np.inf
        self.observation_space = spaces.Box(0.0,  #-high_state
                                             high_state, 
                                             shape=(10,),  
                                             dtype='float32')

        self.action_space = spaces.Box(low=-1.0, 
                                       high=1.0, 
                                       shape=(2,), 
                                       dtype='float32')
    
    def connect(self,head):
        if head:
            # print("direct")
            p.connect(p.DIRECT)
        else:
            # print("GUI")
            p.connect(p.GUI)

    def step(self,action):
        #print('action: ', action)       
        #cont = 550 #15.25 #
        # self.action_scale = np.array([195,95]) #[.675,.975] #based on statistics (determine.py) 
        self.action_scale = np.array([205,100]) #[0.7,0.95]
        self.step_id += 1
        self.jointTorque = np.array(action)*self.action_scale
        #print('jointTorque:',self.jointTorque)

        self.contact_points = p.getClosestPoints(self.objectID_obs,self.armID,10.0)

        ## Torque Control
        self.move(self.jointTorque)
        self.simulate(1)
        state = self.compute_state()
        return state, \
               self.reward(action,state), \
               self.is_done(state), \
               0, \
               {'distance': self.distance(state),
                'step': self.step_id}

    def reset(self):
        self._reset_world()
        
        #self.radius = .445 #[0.2,1.9]
        #self.theta_pos = 31*(np.pi/180) #[0,360] 

        self.radius_obs = 1.25 #0.2 #.445 #[0.35,1.51]
        self.theta_obs = 31*(np.pi/180) #[0,360] 
        x_obs = self.radius_obs*np.cos(self.theta_obs)
        y_obs = self.radius_obs*np.sin(self.theta_obs)

        self.radius = np.random.uniform(low = 0.35, high = (1.51) ) #low = 0.31 | high = l1+l2 - 3*0.05 + 0.01
        self.theta_pos = np.random.uniform(low = 0, high = 2*np.pi)
        x_pos = self.radius*np.cos(self.theta_pos)
        y_pos = self.radius*np.sin(self.theta_pos)
        self.target_coordinates = [x_pos,y_pos]
        """
        objectPos = [x_pos,y_pos,0.15]  
        objectOrn = p.getQuaternionFromEuler([0,0,0])      
        self.objectID = p.loadURDF("sphere_small.urdf",
                    objectPos,
                    objectOrn,
                    useFixedBase = 1,
                    globalScaling = 0.9)
        """
        objectPos = [x_pos,y_pos,-0.05]
        objectOrn = p.getQuaternionFromEuler([0,np.pi/2,0])
        self.objectID = p.loadURDF("block.urdf",
                    objectPos,
                    objectOrn,
                    useFixedBase = 1,
                    globalScaling = 0.9)    

        objectPos_obs = [x_obs,y_obs,0.05]
        objectOrn_obs = p.getQuaternionFromEuler([0,np.pi/2,0])
        self.objectID_obs = p.loadURDF("block.urdf",
                    objectPos_obs,
                    objectOrn_obs,
                    useFixedBase = 1,
                    globalScaling = 0.9)          
        self.simulate(50)
        state = self.compute_state() 

        # print('reset_state: ', state)

        # #new_state presentation (gynamsium)
        # z = np.array(state, dtype='f')
        # issue = (z,{})
        # return issue
        return state

    def simulate(self, num):
        for _ in range(num):
            p.stepSimulation()
            p.performCollisionDetection()

    def _reset_world(self):
        p.setAdditionalSearchPath(pybullet_data.getDataPath()) #optionally 
        p.resetSimulation()
        p.setGravity(0,0,-10)
        p.setTimeStep(0.02) #10ms [affects action frequency] 
        #p.setTimeStep(1) #1ms
        p.setRealTimeSimulation(0)
        planeId = p.loadURDF("plane.urdf")
        armStartPos = [0.0, 0.0, 0.025]
        armStartOrn = p.getQuaternionFromEuler([0.0, 0.0, 0.0])
        self.armID = p.loadURDF(arm_dir,#arm directory
                    armStartPos,
                    armStartOrn,
                    useFixedBase = 1,)
                
        joint_type_name = ['REVOLUTE', 'PRISMATIC', 'SPHERICAL', 'PLANAR', 'FIXED']
        self.joints = []
        self.links = {}
        num_joints = p.getNumJoints(self.armID)
        #print('num_joints: ', num_joints)

        for joint_idx in range(num_joints):
            info = p.getJointInfo(self.armID,joint_idx)
            data = {
                'joint_idx' : info[0],
                'joint_name' : info[1].decode('utf-8'),
                'joint_type' : joint_type_name[ info[2] ],
                'link_name' : info[12].decode('utf-8'),}

            if data['joint_type'] != 'FIXED':
                self.joints.append(data)
                self.links[data['link_name']] = joint_idx

        #print('joints: ', self.joints)
        #print('links: ',self.links)
        self.step_id = 0
        self.distance_threshold = 0.05 #5cm precision **test 2.5cm**

        #velocity_control prior state reset
        for key in self.links:
            p.setJointMotorControl2(self.armID,
                            jointIndex = self.links[key],
                            controlMode = p.VELOCITY_CONTROL,
                            force = 0.0 )

        resetPos = np.random.uniform(0, 2*np.pi, 2 ) #random start pose
        #resetPos = [0, 0]
        for key in self.links:
            p.resetJointState(self.armID,
                        jointIndex = self.links[key],
                        targetValue = resetPos[ self.links[key] ], #selects index
                        targetVelocity = 0.0 )
    
    def move(self,jointInput):
        jointValues = [None]*2
        for idx,val in enumerate(jointInput):
            #print(idx,val)
            jointValues[idx] = val
        p.setJointMotorControlArray(self.armID,
                        jointIndices = [0,1],
                        controlMode = p.TORQUE_CONTROL,
                        forces = jointValues,
                        #forces = jointInput,
                        )
        self.simulate(50)

    def convert(self, angle):        
        angle = np.round(angle,2)
        return angle % (2*np.pi)

    def effector_pos(self,state):
        x = state[6]
        y = state[7]  
        return x,y

    def distance(self,state,type=1):
        #print('state_size: ', len(state))
        if type == 0: #using angles
            angle_vector = np.array(self.IK())
            er1 = angle_vector[0] - angle_vector[2] #O1g - O1 
            er2 = angle_vector[1] - angle_vector[3] #O2g - O2
            dis = np.sqrt( er1**2 + er2**2 ) #w/ IK
        else: #using coordinates
            dis = np.linalg.norm([ state[8],state[9] ]) #w/o IK
        return dis
        
    def compute_state(self):
        jointPos = [None]*2
        jointVel = [None]*2
        joint_states = p.getJointStates(self.armID, jointIndices = [0,1])
        #print(joint_states)
        
        for i in [0,1]:    
            jointPos[i] = self.convert( joint_states[i][0] ) #joint_states[i][0]
            jointVel[i] = joint_states[i][1] 
        s0 = np.cos( jointPos[0] )
        s1 = np.cos( np.sum(jointPos) )
        s2 = np.sin(jointPos[0])
        s3 = np.sin( np.sum(jointPos) )
        s4 = self.radius*np.cos(self.theta_pos) #goal_coord_x
        s5 = self.radius*np.sin(self.theta_pos) #goal_coord_y
        l1 = 0.7 #0.675
        l2 = 0.95 #0.975
        s6 = l1*s0 + l2*s1 #end_effector_coord_x
        s7 = l1*s2 + l2*s3 #end_effector_coord_y
        s8 = s6 - s4 #xe - xg
        s9 = s7 - s5 #ye - yg

        # """        
        s10 = self.radius_obs*np.cos(self.theta_obs) #obstacle_coord_x
        s11 = self.radius_obs*np.sin(self.theta_obs) #obstacle_coord_y
        s12 = 0.5*l1*s0 #link1_CM_coord_x 
        s13 = 0.5*l1*s2 #link1_CM_coord_y
        s14 = l1*s0 + 0.5*l2*s1 #link2_CM_coord_x 
        s15 = l1*s2 + 0.5*l2*s3 #link2_CM_coord_y

        s16 = np.abs(s12 - s10) #xcm1 - xobs
        s17 = np.abs(s13 - s11) #ycm1 - yobs
        s18 = np.abs(s14 - s10) #xcm2 - xobs
        s19 = np.abs(s15 - s11) #ycm2 - yobs

        s20 = l1*s0 #link1_tip_coord_x
        s21 = l1*s2 #link1_tip_coord_y

        self.obstacle_variables = np.array([s16,s17,s18,s19])
        self.min_dis_2_obs = self.obstacle_function([s10,s11,s20,s21,s6,s7])
        # """
        #s10 = jointVel[0]
        #s11 = jointVel[1]
        #print('goal_pos: [{},{}] '.format(s4,s5) )
        #print('end_pos: [{},{}] '.format(s6,s7) )
        #print('distance: ', np.sqrt(s8**2 + s9**2) )
        #print('diff_pos: [{},{}] '.format(s8,s9) )
        return s0,s1,s2,s3,s4,s5,s6,s7,s8,s9
    
    def obstacle_function(selof,vector):
        #coordinates
        o = np.array([vector[0],vector[1]]) #obs_coord
        a = np.array([vector[2],vector[3]]) #link_1_tip_coord
        b = np.array([vector[4],vector[5]]) #link_2_tip_coord

        #sample_points_along_arm
        N = 50

        dmin = 100 #instantiation
        obs_radial_width = 0.02 #2cm [obstacle radius]
        link_thickness = 0.05 #5cm 
        for k in range(N+1):
            pk = a + (k/N)*(b-a)
            dopk = np.linalg.norm(pk-o) - (obs_radial_width+0.5*link_thickness)

            if dopk < dmin:
                dmin = dopk

        # print('dmin: ', dmin)
        return dmin

    def reward(self, jointInput, state, dense=True):
        distance = self.distance(state)
        TorqueControl = np.square(jointInput).sum() 
        #VelControl = np.square([state[6],state[7]]).sum() 
        #print('distance: ',distance)
        #print('TorqueControl: ',TorqueControl)

        #obstacle_reward
        d1 = self.contact_points[1][8] #link_1_distance_to_obstacle
        d2 = self.min_dis_2_obs #link_2 distance_to_obstacle
        a = 2.5 #tuning_factor
        reward_collision = 0 #reward_for_collision
        reward_obstacle =  np.exp(-a*d1) + np.exp(-a*d2) #reward_for_approaching_obstacle
        self.collision = 0

        # print('r_d2: ', 1/(d1 + 0.005))
        # print('r_d1: ', 1/(d2 + 0.005))
        # print('sr_d2: ', np.exp(-a*d1))
        # print('sr_d1: ', np.exp(-a*d2))
        # print('reward_obstacle: ', reward_obstacle)
        
        if d1 <= 0.005 or d2 <= 0.005: #checking_collision
            # print('\n collision')
            # print('d1: ', d1)
            # print('d2: ', d2)
            self.collision = 1
            reward_collision = 1 #if collision occurred
            # xxx
        # print('reward_collision: ', reward_collision)
        if dense:
            return -( distance**2 + TorqueControl + 1000*reward_collision + 5*reward_obstacle)
        else:
            return -(distance < self.distance_threshold).astype(np.float32) 
    
    def is_done(self,state):
        distance = self.distance(state)
        # print('self.collision: ', self.collision)
        # if distance < self.distance_threshold: print('distance: ', distance)
        return self.step_id == 50 or distance < self.distance_threshold or self.collision == 1

    def IK(self):
        jointPos = [None]*2
        joint_states = p.getJointStates(self.armID, jointIndices = [0,1])
        #print(joint_states,)
        
        for i in [0,1]:    
            jointPos[i] = self.convert(joint_states[i][0]) #joint_states[i][0]
        goal_angles = InvK(self.theta_pos, self.radius).compute()
        state_angles = np.array( [jointPos[0], jointPos[1]] )       
        return *goal_angles, *state_angles

    def render(self):
        pass

#Execution
if __name__ == '__main__':
    sim = DoubleLink(0)
    obs = sim.reset()

    action = np.array([0,1])
    state,_,_,_,_ = sim.step(action)
    # print(sim.step(action))
    # print('state: ',state)

    for _ in range(10000000000):
        action = np.array([1,0])
        state,_,_,_,_ = sim.step(action)
        time.sleep(0.1)
    """
    for _ in range(10000000000):    
        action = np.array([1,0])
        #print('action: ',action)
        state,_,_,_ = sim.step(action)
        #print('state: ', state )
        print('states(deg): ', np.round( np.array(sim.IK())*(180/np.pi) ,3) )
        distance = sim.distance(state)
        print('distance: ', distance)
        #print('outward: ', outward)
        #print('______________________________')
        time.sleep(0.01)
        print(sim.theta2)
    """
