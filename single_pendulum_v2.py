import scipy.integrate as spi
import matplotlib.pyplot as plt
import numpy as np
from second_order import second_order



class pendulum(second_order):
    def __init__(self):
        super().__init__(self,wn,zeta,time_step=0.001,input=0, ydotdot = 0, ydot = 0, y = 0)


    def update_ydotdot(self):
        self.ydotdot = self.input - 2*self.zeta*self.wn*self.ydot - np.power((self.wn),2)*np.sin(self.y)


class noisy_pendulum(pendulum):
    def __init__(self):
        super().__init__(self,wn,zeta,time_step=0.001,input=0, ydotdot = 0, ydot = 0, y = 0)

    def step(self):
        self.update_ydotdot()


        self.ydot += self.integration(self.ydotdot)[0]
        self.y += self.integration(self.ydot)[0] # + noise

        self.t += self.dt
