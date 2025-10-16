import numpy as np
from numpy import random
from dynamics import ForwardDynamics as FD

joint_acc = np.array([15,15])
#joint_vel = np.array([4,4])
#joint_pos = np.array([0,0])
collection = []
link1_torques = []
link2_torques = []
for p in range(100000):
	joint_acc = (2*15)*random.random_sample((1,)) - 15 #[-15,15] (rad/s**2)
	joint_vel = (2*np.pi)*random.random_sample((1,)) - np.pi #[-pi,pi] (rad/s)
	joint_pos = (2*np.pi)*random.random_sample((1,))  #[0,2*pi] (rad)
	torques_est = FD(joint_acc,joint_vel,joint_pos).compute()
	link1_torques.append(torques_est)
	# link2_torques.append(torques_est[1])
	# print('torques_est: ', torques_est)
	#print('torque_ratio: ', torques_est[0]/torques_est[1])
print('link1_max_torque: ', np.max(link1_torques))
# print('link2_max_torque: ', np.max(link2_torques))