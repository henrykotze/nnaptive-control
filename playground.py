import numpy as np
import scipy.integrate as spi
import matplotlib.pyplot as plt


def df(t):
    return np.r_[x-2]


t = np.linspace(0,5,100)



v = spi.odeint(df,0,t)







plt.plot(v,t, 'o-', mew=1, ms=8,mec='w')
plt.show()
