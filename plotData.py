#!/usr/bin/env python3
from uav_model import drone
import numpy as np
import matplotlib.pyplot as plt
import sys


filename = './learning_data/response-0.npz'
try:
    responseDesired = sys.argv[1]
    filename = filename.replace("0",responseDesired)
except IndexError:
    print('Provide a number to plot a response')


print("Plotting Response of file: ", filename)


data = np.load(filename)

states = data['features']





plt.rc('text', usetex=True)
plt.rc('font', family='serif')
plt.rc('font', size=12)

plt.figure(1)
plt.plot(states[:,0:4],'-', mew=1, ms=8,mec='w')
plt.legend(['$input$','$\ddot y$','$\dot y $','$y$'])
plt.grid()


# plt.figure(7)
# plt.plot(states[:,4:7],'-', mew=1, ms=8,mec='w')
# plt.legend(['$x$','$y$','$z$'])
# plt.grid()
#
#
# plt.figure(2)
# plt.plot(states[:,7:10],'-', mew=1, ms=8,mec='w')
# plt.legend(['U','V','W'])
# plt.grid()
#
#
#
# plt.figure(3)
# plt.plot(states[:,10:13],'-', mew=1, ms=8,mec='w')
# plt.legend(['$\dot U$','$\dot V$','$\dot W$'])
# plt.grid()
#
#
# plt.figure(4)
# plt.plot(states[:,13:16],'-', mew=1, ms=8,mec='w')
# plt.legend(['$\dot P$','$\dot Q$','$\dot R$'])
# plt.grid()
#
#
#
# plt.figure(5)
# plt.plot(states[:,16:19],'-', mew=1, ms=8,mec='w')
# plt.legend(['$P$','$Q$','$R$'])
# plt.grid()
#
#
# plt.figure(6)
# plt.plot(states[:,19:22],'-', mew=1, ms=8,mec='w')
# plt.legend([r'$\theta$','$\phi$','$\psi$'])
# plt.grid()



# plt.figure(3)
# plt.plot(states[:,6:9],'o-', mew=1, ms=8,mec='w')
# plt.legend([r'$\theta$','$\phi$','$\psi$'])
# plt.grid()


#plt.legend([r'$\dot \theta$','$\dot \phi$','$\dot \psi$'])



plt.show()


data.close()
