import numpy as np
import argparse


"""
execution: python Inverse_Kinematics.py -r <value> -a <value> -o1 <value> -o2 <value>
e.g. python Inverse_Kinematics.py -r 0.5 -a 2.45 -o1 1.24 -o2 .24
"""


parser = argparse.ArgumentParser()
parser.add_argument("-r","--radius", type=float, default=1.0)
parser.add_argument("-a","--angle", type=float, default=1.0)
parser.add_argument("-o1","--theta1", type=float, default=1.0)
parser.add_argument("-o2","--theta2", type=float, default=1.0)
args = parser.parse_args()

def convert(angle):        
    #angle = np.round(angle,2)
    return angle % (2*np.pi)

def main():
    theta_1 = args.theta1
    theta_2 = args.theta2
    r = args.radius
    a = convert(args.angle)

    xe = r*np.cos( a )
    ye = r*np.sin( a )

    beta = np.arctan(xe/ye) - theta_1
    l2 = r*( np.sin(beta)/np.sin(theta_2) )
    l1 = l2*( np.sin(theta_2)/np.tan(beta) - np.cos(theta_2) )

    print('radius: ',r)
    print('beta: ', beta)
    print('l1 = ',l1)
    print('l2 = ',l2)

if __name__ == "__main__":
    main()