import gym

import numpy as np


'''
sys_const: systems constants
Type: np.array

init_cond: initial conditions
'''
class drone():
    g = 9.81 # Gravity constant
    def __init__(self, sys_const, init_cond):
        self.m = sys_const[1] # Mass of drone
        self.Ixx = sys_const[2]
        self.Iyy = sys_const[3]
        self.Izz = sys_const[4]
        self.sys_const = sys_const
        self.x = 0
        self.y = 0
        self.z = 0





# Direct Cosine Matrix to execute axis system transformation through all three
# Euler angles
    def DCM_transfrom(theta,phi,psi):
        return np.matrix([ [np.cos(psi)*np.cos(theta), np.sin(psi)*np.cos(theta), -1*np.sin(theta)], \
                            [np.cos(psi)*np.sin(theta)*np.sin(phi)-np.sin(psi)*np.cos(phi), np.sin(psi)*np.sin(theta)*np.sin(phi)+np.cos(psi)*np.cos(phi), np.sin(psi)*np.sin(theta)*np.cos(phi)-np.cos(psi)*np.sin(phi)], \
                            [-np.sin(theta), np.cos(theta)*np.sin(phi), np.cos(theta)*np.cos(phi)] ])

# Transpose of the Direct Cosine Matrix
    def transpose_DCM(theta,phi,psi):
        return np.transpose(DCM_transfrom(theta,phi,psi))


# Returns the dynamic drag in Newton
# p: density of the fluid
# v: linear velocity of the object to the fluid
# Cd: Drag Coefficient
# A: Reference Area
    def aerodrag(p, v, Cd, A):
        return 0.5*p*np.square(v)*Cd*A

# Model for the wind disturbances
    def wind():
        pass

# Velocity in the X direction
    def dXdt(self):
        dXdt = -w*q+v*r+1/self.m*self.x
        return dXdt

# Velocity in the Y direction
    def dYdt(self):
        dYdt = w*-u*r+1/self.x*self.y
        return dYdt

# Velocity in the Z direction
    def dZdt(self):
        dZdt = -1*v*p+u*q+1/self.m*self.z

    def theta_dot(self):
        theta_dot = np.divide(-1,self.Ixx)*q*r*(self.Izz-self.Iyy)+np.divide(1,self.Ixx)*l





# Determine the next state conditions
    def step():
        pass
