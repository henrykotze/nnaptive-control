#!/usr/bin/env python3
# from uav_model import drone
import numpy as np
import matplotlib.pyplot as plt



t = np.arange(0,300)

i = np.pi*2*2*t/300

y = 3*np.sin(i)

plt.figure(1)
plt.plot(y,'.', mew=1, ms=8,mec='w')
plt.show()
