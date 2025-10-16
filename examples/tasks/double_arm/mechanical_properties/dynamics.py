import numpy as np


class ForwardDynamics():
	def __init__(self,joint_acc,joint_vel,joint_pos):
		self.joint_acc = joint_acc #vector
		self.joint_vel = joint_vel #vector
		self.joint_pos = joint_pos #vector
	
	def compute(self): #see arm.urdf for properties
		l1 = 0.7 #joint-to-joint: 0.65 & full_length: 0.7 
		l2 = 0.95 #joint-to-joint: 0.95 & full_length: 1.0 
		z1 = .5*l1
		z2 = .5*l2
		M1 = 5.0625
		M2 = 6.750
		
		m11 = M1*z1**2 + M2*(l1**2 + z2**2 + 2*l1*z2*np.cos(self.joint_pos[1]) )
		m12 = M2*(z2**2 + l1*z2*np.cos(self.joint_pos[1]) )
		m21 = m12
		m22 = M2*z2**2
		Mass_matrix = np.array([[m11,m12],[m21,m22]])
		
		v11 = -2*M2*l1*z2*np.sin(self.joint_pos[1])*self.joint_vel[1]
		v12 = -M2*l1*z2*np.sin(self.joint_pos[1])*self.joint_vel[1]
		v21 = M2*l1*z2*np.sin(self.joint_pos[1])*self.joint_vel[0]
		v22 = 2*M2*l1*z2*np.sin(self.joint_pos[1])*self.joint_vel[0]
		CC_matrix = np.array([[v11,v12],[v21,v22]])
		
		return Mass_matrix@self.joint_acc + CC_matrix@self.joint_vel
