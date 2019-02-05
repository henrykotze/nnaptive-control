import gym

import numpy as np


class drone(core.Env):
'''
sys_const: systems constants
Type: np.array

init_cond: initial conditions



'''
    __init__(self, sys_const, init_cond):
        self.mass = sys_const[1]
        self.Ixx = sys_const[2]
        self.Iyy = sys_const[3]
        self.Izz = sys_const[4]
        self.sys_const = sys_const



'''
transform a velocity vector in the inertia axis system to the body axis
'''
    def DCM_transfrom(theta,phi,psi,x0,y0,z0):
        return np.matrix([np.cos(psi)*np.cos(theta)], np.sin(psi)*np.cos(theta), -1*np.sin(theta);
                            np.cos(psi)*np.sin(theta)*np.sin(phi)-np.sin(psi)*np.cos(phi), np.sin(psi)*np.sin(theta)*np.sin(phi))+np.cos(psi)*np.cos(phi), np.sin(psi)*np.sin(theta)*np.cos(phi)-np.cos(psi)*np.sin(phi);
                            -np.sin(theta), np.cos(theta)*np.sin(phi), np.cos(theta)*np.cos(phi)])*np.matrix([x0;y0;z0])
