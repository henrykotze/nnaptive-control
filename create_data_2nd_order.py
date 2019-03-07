from second_order import second_order
import numpy as np
import scipy.integrate as spi
import matplotlib.pyplot as plt
import random as rand
import argparse



parser = argparse.ArgumentParser(\
        prog='create data 2nd order',\
        description='Creates .npz files of step responses with given damping ratio\
                    frequency'
        )


parser.add_argument('-zeta', default = 1, help='the damping ratio the response, default: 1')
parser.add_argument('-wn', default= 2, help='the natural frequency of the response, default: 2')
parser.add_argument('-loc', default='./learning_data/', help='location to store responses, default: ./')
parser.add_argument('-filename', default="response-0.npz", help='filename, default: response-*')
parser.add_argument('-t', default=1000, help='time lenght of responses, default: 1000ms')
parser.add_argument('-numSim', default=2, help='number of responses to generate, default: 1')


args = parser.parse_args()

zeta=float(vars(args)['zeta'])
wn=float(vars(args)['wn'])
time=int(vars(args)['t'])
numberSims = int(vars(args)['numSim'])
dir = vars(args)['loc']
filename = vars(args)['loc']+'/'+vars(args)['filename']


# Add a Readme file in directory to show selected variables that describe the
# responses
info = '*******************************'+'\n'+'INFORMATION'+'\n'+\
        '*******************************'+'\n'+'wn: '+str(wn)+'\n'\
        +'zeta: '+str(zeta)+'\n'+'time: ' + str(time) \
        + '\n' + 'number of responses: ' + str(numberSims)

f = open(str(dir + '/readme.txt'), 'w+')

f.write(info)
f.close()




# filename = './test_data/response-0.npz'

max_input=5
# Minimum thrust delivered
min_input=-5

init_condition_max = 1
init_condition_min = -1




if __name__ == '__main__':
    print('Writing responses to: ', filename)
    for numSim in range(0,numberSims):
        print('Number of simulation: ', numSim)

        response = second_order(wn,zeta)
        response.update_input(rand.uniform(min_input,max_input)/10)

        input = np.zeros(4)
        output = np.zeros(3)


        for t in range(0,time):

            input = np.vstack( (input, response.getAllStates()+np.random.randn(1,4)*0) )
            response.step()
            output = np.vstack( (output, response.getEstimatedStates() ) )


        # Saves response in *.npz file
        np.savez(filename,features=input,labels=output)

        # Change number on filename to correspond to simulation number
        filename = filename.replace(str(numSim),str(numSim+1))
