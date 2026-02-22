import numpy as np

def convert(angle):
            print('angle: ', angle)
            if angle == 0.0: sign = 0.0
            else: sign = np.sin(angle)/np.abs(np.sin(angle))
            angle = angle % np.pi 
            if sign < 0: angle = angle - np.pi
            return angle

r = float( input("Radius: ") )
a = float( input("Angle(deg): ") )

a = a*(np.pi/180)
a = convert(a)
print('angle_tr: ', a*(180/np.pi))

xe = r*np.cos( a )
ye = r*np.sin( a )

l1 = 0.9
l2 = 0.85

o2p = np.arccos( (xe**2 + ye**2 - l1**2 -l2**2 )/(2*l1*l2) )
o2n = -np.arccos( (xe**2 + ye**2 - l1**2 -l2**2 )/(2*l1*l2) )

if (-np.pi/2 < a < np.pi/2):
    o1p = np.arctan(ye/xe) - np.arctan( (l2*np.sin(o2p))/(l1+l2*np.cos(o2p)) )
    o1n = np.arctan(ye/xe) + np.arctan( (l2*np.sin(o2n))/(l1+l2*np.cos(o2n)) )
else:
    o1p = np.pi + ( np.arctan(ye/xe) - np.arctan( (l2*np.sin(o2p))/(l1+l2*np.cos(o2p)) ) )
    o1n = np.pi + ( np.arctan(ye/xe) + np.arctan( (l2*np.sin(o2n))/(l1+l2*np.cos(o2n)) ) )


angle_neg = np.array([o1n, o2n])
angle_pos = np.array([o1p, o2p])

print('For o2 > 0 (rad): {} '.format( angle_pos) )
print('For o2 < 0 (rad): {} '.format( angle_neg) )
print('For o2 > 0 (deg): {} '.format( angle_pos*(180/np.pi) ))
print('For o2 < 0 (deg): {} '.format( angle_neg*(180/np.pi) ))

