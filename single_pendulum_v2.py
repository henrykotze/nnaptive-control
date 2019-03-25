import scipy.integrate as spi
import matplotlib.pyplot as plt
import numpy as np
from second_order import second_order



class pendulum(second_order):

    def update_ydotdot(self):
        self.ydotdot = self.input - 2*self.zeta*self.wn*self.ydot - np.power((self.wn),2)*np.sin(self.y)


class noisy_pendulum(pendulum):

    def noise(self,lower, upper):
        return (upper-lower)*np.random.random(1)+lower

    def step(self):
        self.update_ydotdot()


        self.ydot += self.integration(self.ydotdot)[0]
        self.y += self.integration(self.ydot)[0] #+ self.noise(-0.00005,0.00005)

        self.t += self.dt
