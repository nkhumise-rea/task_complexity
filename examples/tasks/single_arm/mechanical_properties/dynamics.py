import numpy as np


class ForwardDynamics():
	def __init__(self,joint_acc,joint_vel,joint_pos):
		self.joint_acc = joint_acc #vector
		self.joint_vel = joint_vel #vector
		self.joint_pos = joint_pos #vector
	
	def compute(self): #see arm.urdf for properties
		l1 = 1.65 #joint-to-joint: 1.65 & full_length: 1.70 
		M1 = 11.4750 
		# l1 = 0.95 #joint-to-joint: 0.95 & full_length: 1.00 
		# M1 = 6.75
		z1 = .5*l1
		

		m = (1/3)*M1*(l1*l1) #moment of inertia

		# print(self.joint_acc)
		# print([m]@self.joint_acc)
		# xxx
				
		return [m]@self.joint_acc 
