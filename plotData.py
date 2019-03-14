#!/usr/bin/env python3
# from uav_model import drone
import numpy as np
import matplotlib.pyplot as plt
import sys
import argparse


parser = argparse.ArgumentParser(\
        prog='Plot Data',\
        description='Creates .npz files of step responses with given damping ratio\
                    frequency'
        )


parser.add_argument('-loc', default='./train_data/', help='location to store responses, default: ./train_data')
parser.add_argument('-filename', default="response-0.npz", help='filename, default: response-0.npz')
parser.add_argument('-n', default=0, help='plot a specific number response')


args = parser.parse_args()

dir = vars(args)['loc']
filename = vars(args)['loc']+'/'+vars(args)['filename']
number = int(vars(args)['n'])

filename = filename.replace(str(0),str(number))
print("Plotting Response of file: ", filename)


data = np.load(filename)

y = data['y_']
ydot = data['y_dot']
ydotdot = data['y_dotdot']
input = data['input']





plt.rc('text', usetex=True)
plt.rc('font', family='serif')
plt.rc('font', size=12)

plt.figure(1)
plt.plot(input,'-', mew=1, ms=8,mec='w')
plt.plot(ydotdot,'-', mew=1, ms=8,mec='w')
plt.plot(ydot,'-', mew=1, ms=8,mec='w')
plt.plot(y,'-', mew=1, ms=8,mec='w')
plt.legend(['$input$','$\ddot y$','$\dot y $','$y$'])
plt.grid()


plt.show()


data.close()
