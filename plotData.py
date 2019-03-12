#!/usr/bin/env python3
from uav_model import drone
import numpy as np
import matplotlib.pyplot as plt
import sys


filename = './train_data/response-0.npz'
try:
    responseDesired = sys.argv[1]
    filename = filename.replace("0",responseDesired)
except IndexError:
    print('Provide a number to plot a response')


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
plt.plot(y,'-', mew=1, ms=8,mec='w')
plt.plot(ydot,'-', mew=1, ms=8,mec='w')
plt.plot(ydotdot,'-', mew=1, ms=8,mec='w')
plt.legend(['$input$','$\ddot y$','$\dot y $','$y$'])
plt.grid()


plt.show()


data.close()
