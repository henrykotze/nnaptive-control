#!/usr/bin/env python3




class PIDcontroller:
    def __init__(self,P,I,D,error=0,time=0,output_control=0,dt=0.01):
        self.P = P
        self.I = I
        self.D = D
        self.e = error

        self.intergrated_error = 0
        self.prev_error = 0

        self.output_control = output_control
        self.t = time
        self.dt = dt


    def integration(self,x):
        from scipy.integrate import quad
        f = lambda t: x
        return quad(f,self.t, self.t+self.dt)

    def updateError(self,error):
        self.e = error


    def outputControl(self):
        self.output_control = self.P*self.e + self.I*self.intergrated_error + self.D*(self.e-self.prev_error)/self.dt
        return self.output_control


    def step(self,error):
        self.prev_error = self.e
        self.e = error
        self.intergrated_error += self.integration(self.e)[0]


class stateSpace():
    def __init__(self,K):
        self.fb_gain = K
