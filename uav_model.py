import gym

import numpy as np


'''
sys_const: systems constants
Type: np.array

init_cond: initial conditions
'''
class drone():
    def __init__(self, sys_const, init_cond):
        self.mass = sys_const[1]
        self.Ixx = sys_const[2]
        self.Iyy = sys_const[3]
        self.Izz = sys_const[4]
        self.sys_const = sys_const



# Direct Cosine Matrix to execute axis system transformation through all three
#Euler angles
    def DCM_transfrom(theta,phi,psi):
        return np.matrix([ [np.cos(psi)*np.cos(theta), np.sin(psi)*np.cos(theta), -1*np.sin(theta)], \
                            [np.cos(psi)*np.sin(theta)*np.sin(phi)-np.sin(psi)*np.cos(phi), np.sin(psi)*np.sin(theta)*np.sin(phi)+np.cos(psi)*np.cos(phi), np.sin(psi)*np.sin(theta)*np.cos(phi)-np.cos(psi)*np.sin(phi)], \
                            [-np.sin(theta), np.cos(theta)*np.sin(phi), np.cos(theta)*np.cos(phi)] ])

# Transpose of the Direct Cosine Matrix
    def transpose_DCM(theta,phi,psi):
        return np.transpose(DCM_transfrom(theta,phi,psi))
