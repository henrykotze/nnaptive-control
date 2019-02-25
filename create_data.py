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
# System Constants
sys_constants = [1,1,1,1]


# base filename
filename = 'response-0.txt'



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
    return [thrust1,thrust2,thrust3,thrust4]

# Writes the data to a file
def writeData():
    pass


# Run the simulation and gather information
def simulate():


    for numSim in range(0,10):
        init_conditions = init_conds()
        drone1 = drone(sys_constants,init_conditions)


        for t in range(0,2):

            drone1.step()
            states = np.vstack( (states, drone1.getStates() ) )
            drone1.setThrust(inputThrust())






states = np.zeros(9)



if __name__ == '__main__':

    for numSim in range(0,2):
        print('Number of simulation: ', numSim)
        file = open(filename, 'w+')

        init_conditions = init_conds()
        drone1 = drone(sys_constants,init_conditions)


        for t in range(0,2):

            drone1.step()
            states = np.vstack( (states, drone1.getStates() ) )
            drone1.setThrust(inputThrust())


        np.savetxt(file,states)
        file.close()

        # Change number on filename to correspond to simulation number
        filename = filename.replace(str(numSim),str(numSim+1))





    pass
