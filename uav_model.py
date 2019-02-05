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
        self.t = 0
        self.dt = 0.1





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

# Returns the components of (m*g)
    def gravity(self):
        np.matrix([-1*np.sin(theta)],[np.cos(theta)*np.sin(phi)],[np.cos(theta)*np.cos(phi)])
        return DCM_transfrom(theta,phi,psi)*np.matrix( [0],[0],[self.m*g] )



# Velocity in the X direction
    def dXdt(self):
        dXdt = -w*q+v*r+1/self.m*self.x
        return dXdt

# Velocity in the Y direction
    def dYdt(self):
        dYdt = w*-u*r+np.divide(1,self.x)*self.y
        return dYdt

# Velocity in the Z direction
    def dZdt(self):
        dZdt = -1*v*p+u*q+np.divide(1,self.m)*self.z
        return dZdt

    def theta_dot(self):
        theta_dot = np.divide(-1,self.Ixx)*q*r*(self.Izz-self.Iyy)+np.divide(1,self.Ixx)*l
        return theta_dot

    def  psi_dot(self):
        psi_dot = np.divide(-1,self.Iyy)*p*r*(self.Ixx-self.Izz)+np.divide(1,self.Iyy)*m
        return psi_dot

    def phi_dot(self):
        phi_dot = np.divide(-1,self.Izz)*p*q*(self.Iyy-self.Ixx)+np.divide(1,self.Izz)*n

    def dThrust_dt(self,tau):


    def sumForces_X(self):
        pass

    def sumForces_Y(self):
        pass

    def sumForces_Z(self):
        pass


    def sumForces_l(self):
        pass

    def sumForces_m(self):
        pass

    def sumForces_n(self):
        pass


    def thrust(self):
        pass


# Determine the next state conditions
    def step():
        pass
