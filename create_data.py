from uav_model import drone
import numpy as np
import scipy.integrate as spi
import matplotlib.pyplot as plt
import random as rand


## Constants

# pi
PI = np.pi
# Maximum thrust delivered
thrust_MAX=30
# Minimum thrust delivered
thrust_MIN=20





# Provides initial conditions to start simulations
def init_conds():
    theta = rand.uniform(-PI,PI)
    phi = rand.uniform(-PI,PI)
    psi = rand.uniform(-PI,PI)




# Provides random inputs to drone
def inputThrust():
    thrust1 = rand.uniform(thrust_MIN,thrust_MAX)
    thrust2 = rand.uniform(thrust_MIN,thrust_MAX)
    thrust3 = rand.uniform(thrust_MIN,thrust_MAX)
    thrust4 = rand.uniform(thrust_MIN,thrust_MAX)

# Writes the data to a file
def writeData():
    pass


# Run the simulation and gather information
def simulate():
    pass




if __name__ == '__main__':
    pass
