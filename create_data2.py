#!/usr/bin/env python3


from second_order import second_order
import numpy as np
import scipy.integrate as spi
import matplotlib.pyplot as plt
import random as rand
import argparse
from single_pendulum_v2 import pendulum, noisy_pendulum
import os
import pickle
import shelve
from scipy.optimize import curve_fit

parser = argparse.ArgumentParser(\
        prog='create data 2nd order',\
        description='Creates .npz files of step responses with given damping ratio\
                    frequency'
        )


parser.add_argument('-zeta', default = 1, help='the damping ratio the response, default: 1')
parser.add_argument('-wn', default= 2, help='the natural frequency of the response, default: 2')
parser.add_argument('-loc', default='./train_data/', help='location to store responses, default: ./train_data')
parser.add_argument('-filename', default="response-0.npz", help='filename, default: response-0.npz')
parser.add_argument('-t', default=10, help='time lenght of responses, default: 10s')
parser.add_argument('-numSim', default=2, help='number of responses to generate, default: 1')
parser.add_argument('-inputMag', default=1, help='magnitude of input given to system, default: +-1')
parser.add_argument('-system', default='pendulum', help='type of system to generate data, default: pendulum')
parser.add_argument('-init', default=0, help='Initial Condition, default: 0')
parser.add_argument('-rand', default=0, help='pick from normal distribution the input, default: 0')
parser.add_argument('-inputTime', default=50, help='time at which inputs starts, default: 50ms')
parser.add_argument('-startNumSim', default=0, help='to start the response-* different then 0, default: 0')
parser.add_argument('-dt', default=0.01, help='timestep increments of responses, default: 0.01')
parser.add_argument('-maxInput', default=0.5, help='maximum input given to system')
parser.add_argument('-minInput', default=-0.5, help='minimum input given to system')
parser.add_argument('-noise', default=0, help='use a noise pendulum system')
parser.add_argument('-randomInput', default=0, help='use a noise pendulum system')



args = parser.parse_args()

zeta=float(vars(args)['zeta'])
wn=float(vars(args)['wn'])
t=int(vars(args)['t'])
startSimNum = int(vars(args)['startNumSim'])
numberSims = int(vars(args)['numSim']) + startSimNum
dir = vars(args)['loc']
filename = vars(args)['loc']+'/'+vars(args)['filename']
system = str(vars(args)['system'])
initial = float(vars(args)['init'])
randomMag = int(vars(args)['rand'])
inputTime = int(vars(args)['inputTime'])
dt = float(vars(args)['dt'])
inputMag = float(vars(args)['inputMag'])
maxInput = float(vars(args)['maxInput'])
minInput = float(vars(args)['minInput'])
noise = int(vars(args)['noise'])
randomInput = int(vars(args)['randomInput'])



if(noise == 1):
    system_info = 'noisy ' + system
else:
    system_info = system

# Add a Readme file in directory to show selected variables that describe the
# responses

filename = filename.replace(str(0),str(startSimNum))


with shelve.open( str(dir+'/readme') ) as db:
    for arg in vars(args):
        db[arg] = getattr(args,arg)
db.close()


def determine_system(system,wn,zeta,initial_condition):
    response = pendulum(wn,zeta,y=initial_condition*np.pi/180,time_step=dt)
    return response

def generateStepInput(responseDuration,startInput,minInput,maxInput):

    input = np.zeros( (responseDuration,1) )
    timestep = startInput

    while timestep < responseDuration:
        magInput = (maxInput-minInput)*np.random.random()+minInput # Magnitude Size of Input
        inputDur = int(responseDuration/10*(np.random.random() ) ) # Duration of input
        zeroInputDur = int(responseDuration/10*(np.random.random()) ) # Duration of zero input


        input[timestep:timestep+inputDur] = magInput
        timestep += inputDur
        input[timestep:timestep+zeroInputDur] = 0
        timestep += zeroInputDur

    input = addNoise(input,250)
    return input



def generateRampInput(responseDuration,startInput,minInput,maxInput):

    input = np.zeros( (responseDuration,1) )
    timestep = startInput

    while timestep < responseDuration:
        magInput = (maxInput-minInput)*np.random.random()+minInput # peak point in ramp
        firstDur = int(responseDuration/10*(np.random.random() ) )+1 # Duration of first half ramp
        secondDur = int(responseDuration/10*(np.random.random()) )+1 # Duration of second half ramp
        if(timestep + firstDur+secondDur < responseDuration):

            grad1 = magInput/firstDur   # gradient of first part
            grad2 = -magInput/secondDur  # Gradientr of second part

            firstLine = np.arange(firstDur)*grad1

            secondLine = -1*np.arange(secondDur,0,-1)*grad2
            input[timestep:timestep+firstDur] = np.transpose(np.array([firstLine]))
            timestep += firstDur
            input[timestep:timestep+secondDur] = np.transpose(np.array([secondLine]))
            timestep += secondDur
        else:
            break

    input = addNoise(input,250)
    return input

def straightline_func(x, a, b):
    return a*x+b

def exponential_func(x, a, b):
    try:
        y= a*np.exp(b*x)
        return y
    except:
        return 0*x

def quadratic_func(x,a):
    return a*np.power(x,2)

def generateAccInput(responseDuration,startInput,minInput,maxInput):
    input = np.zeros( (responseDuration,1) )
    timestep = startInput

    while timestep < responseDuration:

        magInput = (maxInput-minInput)*np.random.random()+minInput # peak point in ramp
        Dur = int(responseDuration/10*(np.random.random()))+10 # Duration of first half ramp
        Dur2 = int(responseDuration/10*(np.random.random()))+10 # Duration of first half ramp
        neg = 1.0

        if(magInput < 0):
            magInput = -1*magInput
            neg = -1.0

        if(timestep + Dur + Dur2+1 < responseDuration):
            y_ = np.sqrt(np.array([0.00001, magInput]))
            x = np.array([timestep+1,timestep+Dur])
            popt, pcov = curve_fit(straightline_func, x, y_)
            a = np.power(popt[0],2)

            try:
                curve = np.arange(timestep,timestep+Dur)
                curve = neg*quadratic_func(curve, a)
                input[timestep:timestep+Dur] = np.transpose(np.array([curve]))
                y_ = np.sqrt(np.array([magInput, 0.001]))
                x = np.array([timestep+Dur,timestep+Dur+Dur2])
                popt, pcov = curve_fit(straightline_func, x, y_)
                a = popt[0]
                curve = np.arange(timestep+Dur,timestep+Dur+Dur2)
                curve = neg*quadratic_func(curve, a)
                input[timestep+Dur:timestep+Dur+Dur2] = np.transpose(np.array([curve]))
                timestep = timestep + Dur+Dur2
            except:
                timestep = timestep + Dur+Dur2
        else:
            break

    input = addNoise(input,250)
    return input

def generateExpoInput(responseDuration,startInput,minInput,maxInput):
    input = np.zeros( (responseDuration,1) )
    timestep = startInput

    while timestep < responseDuration:

        magInput = (maxInput-minInput)*np.random.random()+minInput # peak point in ramp
        Dur = int(responseDuration/20*(np.random.random()))+10 # Duration of first half ramp
        Dur2 = int(responseDuration*(np.random.random()))+20 # Duration of first half ramp
        neg = 1.0

        if(magInput < 0):
            magInput = -1*magInput
            neg = -1.0

        if(timestep + Dur + Dur2+1 < responseDuration):

            try:
                y_ = np.log(np.array([0.01, magInput]))
                x = np.array([timestep+1,timestep+Dur])
                popt, pcov = curve_fit(straightline_func, x, y_)
                b = popt[0]
                a = np.exp(popt[1])

                curve = np.arange(timestep,timestep+Dur)
                curve = neg*exponential_func(curve, a, b)

                input[timestep:timestep+Dur] = np.transpose(np.array([curve]))

                y_ = np.log(np.array([magInput, 0.01]))
                x = np.array([timestep+Dur,timestep+Dur+Dur2])
                popt, pcov = curve_fit(straightline_func, x, y_)
                b = popt[0]
                a = np.exp(popt[1])
                curve = np.arange(timestep+Dur,timestep+Dur+Dur2)
                curve = neg*exponential_func(curve, a, b)
                input[timestep+Dur:timestep+Dur+Dur2] = np.transpose(np.array([curve]))
                timestep = timestep + Dur+Dur2
            except:
                print("meeeeeeeeeeeeeeeeeeeeeeeep")
                timestep = timestep + Dur+Dur2

        else:
            break
    input = addNoise(input,250)
    return input

def generateNoiseInput(responseDuration,startInput,minInput,maxInput):

    input = np.zeros( (responseDuration,1) )
    input += (maxInput-minInput)*np.random.random((np.size(input),1))+minInput
    return input

def addNoise(response,level):
    sizeOfArray = np.size(response)
    response += np.random.random((sizeOfArray,1))/level
    return response

def generateCombinationInput(responseDuration,startInput,minInput,maxInput):
    input1 = generateStepInput(responseDuration,startInput,minInput/3,maxInput/3)
    input2 = generateRampInput(responseDuration,startInput,minInput/3,maxInput/3)
    input3 = generateExpoInput(responseDuration,startInput,minInput/3,maxInput/3)
    input = addNoise(input1+input2+input3,250)
    return input



if __name__ == '__main__':
    print('Creating the response of ', str(system_info))
    print('Writing responses to:', filename )
    print(startSimNum, numberSims)

    timeSteps= int(t/dt) # time in number of step


    for numSim in range(startSimNum,numberSims):
        print('Number of responses: ', numSim)
        response = determine_system(system,wn,zeta,initial)

        # input_type = np.random.randint(0,3)
        # if(input_type == 0):
        #     input = generateStepInput(timeSteps,inputTime,minInput,maxInput)
        # elif(input_type == 1):
        #     input = generateRampInput(timeSteps,inputTime,minInput,maxInput)
        # elif(input_type == 2):
        #     input =  generateCombinationInput(timeSteps,inputTime,minInput,maxInput);
        # elif(input_type == 3):
        #     input = generateNoiseInput(timeSteps,inputTime,minInput,maxInput)

        input = generateCombinationInput(timeSteps,inputTime,minInput,maxInput)
        y = np.zeros( (timeSteps,1) )
        ydot = np.zeros( (timeSteps,1) )
        ydotdot = np.zeros( (timeSteps,1) )

        for t in range(0,timeSteps):
            # time at which input starts
            if(t == inputTime and randomInput == 0):
                if(randomMag == 0):
                    response.update_input(inputMag)
                else:
                    response.update_input(np.random.uniform(minInput,maxInput))

            elif(randomInput == 1):
                response.update_input( input[t] )


            # temporary variables
            t1,t2,t3,t4 = response.getAllStates()
            input[t] = t1
            y[t] = t4
            ydot[t] = t3
            ydotdot[t] = t2

            # next time step
            response.step()


        if(noise == 1):
            y = addNoise(y,500)

        # Saves response in *.npz file
        # print(system)
        np.savez(filename,input=input,y_=y,y_dot=ydot,y_dotdot=ydotdot,zeta=zeta,wn=wn,system=str(system_info))

        # Change number on filename to correspond to simulation number
        filename = filename.replace(str(numSim),str(numSim+1))
