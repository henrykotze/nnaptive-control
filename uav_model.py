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
        # Notation used from Paul D Moller Thesis
        self.m = sys_const[1] # Mass of drone
        self.Ixx = sys_const[2]
        self.Iyy = sys_const[3]
        self.Izz = sys_const[4]
        self.sys_const = sys_const

        # Position
        self.X = 0  # Force magnitude
        self.Y = 0  # Force magnitude
        self.Z = 0  # Force magnitude

        # Moment
        self.L = 0  # Rolling Moment
        self.M = 0  # Pitching Moment
        self.N = 0  # Yawing Moment

        # Velocity
        self.U = 0  # Velocity in X direction
        self.V = 0  # Velocity in Y direction
        self.W = 0  # Velocity in Z direction

        # Angulare Velocity
        self.P = 0  # Roll Rate
        self.Q = 0  # Pitch Rate
        self.R = 0  # Yaw Rate

        # Thrust of each motor
        self.T1 = 0
        self.T2 = 0
        self.T3 = 0
        self.T4 = 0

        self.t = 0  #
        self.dt = 0.1


        self.solver = scipy.integrate.RK4()






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



# Acceleration in the X direction
    def Udot(self):
        Udot = np.divide(self.X, self.m) + self.V*self.R - self.W*self.Q
        return Udot

# Acceleration in the Y direction
    def Vdot(self):
        Vdot = np.divide(self.Y, self.m) - self.U*self.R + self.W*self.P
        return Vdot

# Acceleration in the Z direction
    def Qdot(self):
        Qdot = np.divide(self.Z, self.m) + self.U*self.Q - self.V*self.P
        return Qdot

    def Pdot(self):
        Pdot = np.divide(self.L, self.Ixx) - np.divide( (self.Izz - self.Iyy), self.Ixx)*self.Q*self.R
        return Pdot

    def Q(self):
        pass

    def Rdot(self):
        Rdot = np.divide(self.N, self.Izz) - np.divide( (self.Iyy - self.Ixx ))*self.P*self.Q
        return Rdot


# Rotor lag dynamics
    def dThrust_dt(self,T,tau,Tr):
        dThrust_dt = np.divide(T,tau) + np.divide(Tr,tau)
        return dThrust_dt



    def sumForces_X(self):
        pass

    def sumForces_Y(self):
        pass

    def sumForces_Z(self):
        self.Z = -1*(self.T1 + self.T2 + self.T3 + self.T4)
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
    def step(self):


        pass
