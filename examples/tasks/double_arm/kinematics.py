import numpy as np

class InverseKinematics():
    def __init__(self,angle, radius):
        self.r = radius
        a = angle #*(np.pi/180)
        self.a = self.convert(a)
        #print('angle_tr: ', self.a*(180/np.pi))

    def compute(self):
        xe = self.r*np.cos( self.a )
        ye = self.r*np.sin( self.a )

        l1 = 0.7 #0.675
        l2 = 0.95 #0.975

        #l1 = 0.65 #joint-to-joint: 0.65 & full_length:0.75 
        #l2 = 0.95 #joint-to-joint: 0.95 & full_length: 1.0 

        o2 = np.arccos( (xe**2 + ye**2 - l1**2 - l2**2 )/(2*l1*l2) )
        o1 = np.arctan(ye/xe) - np.arctan( (l2*np.sin(o2))/(l1 + l2*np.cos(o2)) )

        """
        if (-np.pi/2 < self.a < np.pi/2):
            o1p = np.arctan(ye/xe) - np.arctan( (l2*np.sin(o2p))/(l1+l2*np.cos(o2p)) )
            #o1n = np.arctan(ye/xe) + np.arctan( (l2*np.sin(o2n))/(l1+l2*np.cos(o2n)) )
        else:
            o1p = np.pi + ( np.arctan(ye/xe) - np.arctan( (l2*np.sin(o2p))/(l1+l2*np.cos(o2p)) ) )
            #o1n = np.pi + ( np.arctan(ye/xe) + np.arctan( (l2*np.sin(o2n))/(l1+l2*np.cos(o2n)) ) )
        """
        return np.array([o1,o2])

    def convert(self, angle):        
        """        
        if angle == 0.0: sign = 0.0
        else: sign = np.sin(angle)/np.abs(np.sin(angle))
        angle = angle % np.pi 
        if sign < 0: angle = angle - np.pi
        """
        angle = np.round(angle,2)
        return angle % (2*np.pi)
