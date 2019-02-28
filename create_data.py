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
thrust_MIN=1
# System Constants
sys_constants = [1,1,1,1]
# time of simulations
time=1000


# base filename
filename = './learning_data/response-0.npz'



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


if __name__ == '__main__':

    for numSim in range(0,100):
        print('Number of simulation: ', numSim)

        init_conditions = init_conds()
        drone1 = drone(sys_constants,init_conditions)
        drone1.setThrust(inputThrust())


        input = np.zeros(22)
        output = np.zeros(18)


        for t in range(0,time):

            input = np.vstack( (input,drone1.getAllStates() ) )
            drone1.step()
            output = np.vstack( (output, drone1.getEstimatedStates() ) )


        # Saves response in *.npz file
        np.savez(filename,features=input,labels=output)

        # Change number on filename to correspond to simulation number
        filename = filename.replace(str(numSim),str(numSim+1))
