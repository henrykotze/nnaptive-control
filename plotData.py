from uav_model import drone
import numpy as np
import matplotlib.pyplot as plt


filename = './learning_data/response-4.npz'



data = np.load(filename)

states = data['features']





plt.rc('text', usetex=True)
plt.rc('font', family='serif')
plt.rc('font', size=12)

plt.figure(1)
plt.plot(states[:,0:4],'-', mew=1, ms=8,mec='w')
plt.legend(['$T_{1}$','$T_{2}$','$T_{3}$','$T_{4}$'])
plt.grid()

plt.figure(2)
plt.plot(states[:,7:10],'-', mew=1, ms=8,mec='w')
plt.legend(['U','V','W'])
plt.grid()



plt.figure(3)
plt.plot(states[:,10:13],'-', mew=1, ms=8,mec='w')
plt.legend(['$\dot U$','$\dot V$','$\dot W$'])
plt.grid()


plt.figure(4)
plt.plot(states[:,13:16],'-', mew=1, ms=8,mec='w')
plt.legend(['$\dot P$','$\dot Q$','$\dot R$'])
plt.grid()



plt.figure(5)
plt.plot(states[:,16:19],'-', mew=1, ms=8,mec='w')
plt.legend(['$P$','$Q$','$R$'])
plt.grid()

# plt.figure(3)
# plt.plot(states[:,6:9],'o-', mew=1, ms=8,mec='w')
# plt.legend([r'$\theta$','$\phi$','$\psi$'])
# plt.grid()


#plt.legend([r'$\dot \theta$','$\dot \phi$','$\dot \psi$'])



plt.show()


data.close()
