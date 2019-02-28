from second_order import second_order
import numpy as np
import scipy.integrate as spi
import matplotlib.pyplot as plt
import random as rand



zeta=0.15
wn=8
time=8000


filename = './test_data/response-0.npz'

max_input=1
# Minimum thrust delivered
min_input=-1





if __name__ == '__main__':

    for numSim in range(0,40):
        print('Number of simulation: ', numSim)

        response = second_order(wn,zeta)
        response.update_input(rand.uniform(min_input,max_input))

        input = np.zeros(4)
        output = np.zeros(3)


        for t in range(0,time):

            input = np.vstack( (input,response.getAllStates() ) )
            response.step()
            output = np.vstack( (output, response.getEstimatedStates() ) )


        # Saves response in *.npz file
        np.savez(filename,features=input,labels=output)

        # Change number on filename to correspond to simulation number
        filename = filename.replace(str(numSim),str(numSim+1))
