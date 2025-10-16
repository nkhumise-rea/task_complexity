import numpy as np

#reference: wikipedia.org/wiki/List_of_moments_of_inertia

#Box
#dimensions kg, m

height = 0.75 #y-direction
depth,width = 0.05,0.05
volume = depth*height*width #[m3]
density = 2700.00 #[kg.m-3]#aluminum
box_mass = density*volume

print("mass: ", box_mass)

#w.r.t Centre of Mass
ixx = ( box_mass * ( depth**2 + height**2 ))/12 #z,y
iyy = ( box_mass * ( width**2 + depth**2 ))/12 #x,z
izz = ( box_mass * ( width**2 + height**2 ))/12 #x,y

print("Box:")
print(f"ixx: {ixx}\n iyy: {iyy}\n izz: {izz}")

# #Prism
# prism_mass = 0.4
# prism_width = 0.05 #x-direction
# prism_height = 0.06 #0.05 #y-direction
# prism_depth = 0.05 #0.10 #z-direction

# ixx = ( prism_mass * ( prism_depth**2 + prism_height**2 ))/12 #z,y
# iyy = ( prism_mass * ( prism_width**2 + prism_depth**2 ))/12 #x,z
# izz = ( prism_mass * ( prism_width**2 + prism_height**2 ))/12 #x,y

# print("Prism:")
# print(f"ixx: {ixx}\n iyy: {iyy}\n izz: {izz}")