import gym

import scipy.integrate as spi
import matplotlib.pyplot as plt
import numpy as np
#from scipy.integrate import quad


'''
sys_const: systems constants
Type: np.array

init_cond: initial conditions
'''
class drone():
    g = 9.81 # Gravity constant
    def __init__(self, sys_const, init_cond=np.array([0,0,0,0,0,0]), time_step=0.001,sim_time=10):
        # Notation used from Paul D Moller Thesis
        self.m = sys_const[0] # Mass of drone
        self.Ixx = sys_const[1]
        self.Iyy = sys_const[2]
        self.Izz = sys_const[3]
        self.sys_const = sys_const

        # Forces
        self.X = 0  # Force magnitude in X direction
        self.Y = 0  # Force magnitude in Y direction
        self.Z = 0  # Force magnitude in Z direction

        # Moment
        self.L = 0  # Rolling Moment
        self.M = 0  # Pitching Moment
        self.N = 0  # Yawing Moment

        # Velocity
        self.U = 0  # Velocity in X direction
        self.V = 0  # Velocity in Y direction
        self.W = 0  # Velocity in Z direction

        # Acceleration
        self.Udot = 0  # Acceleration in X direction
        self.Vdot = 0  # Acceleration in Y direction
        self.Wdot = 0  # Acceleration in Z direction

        # Angulare Velocity
        self.P = 0  # Roll Rate
        self.Q = 0  # Pitch Rate
        self.R = 0  # Yaw Rate

        # Angulare Acceleration
        self.Pdot = 0  # Roll Acceleration
        self.Qdot = 0  # Pitch Acceleration
        self.Rdot = 0  # Yaw Acceleration

        # Thrust of each motor
        self.T1 = 0
        self.T2 = 0
        self.T3 = 0
        self.T4 = 0

        self.t = 0  #
        self.dt = time_step
        self.sim_time = sim_time


        self.d = 0.5

        # Position in Cartesian plane
        self.xPos = 0
        self.yPos = 0
        self.zPos = 0


        self.theta_dot = 0
        self.phi_dot = 0
        self.psi_dot = 0

        # Body angles of Drone
        self.theta = 0
        self.phi = 0
        self.psi = 0


        self.error = 0





        #self.solver = scipy.integrate.RK4()






# Direct Cosine Matrix to execute axis system transformation through all three
# Euler angles
    def DCM_transfrom(theta,phi,psi):
        return np.matrix([ [np.cos(psi)*np.cos(theta), np.sin(psi)*np.cos(theta), -1*np.sin(theta)], \
                            [np.cos(psi)*np.sin(theta)*np.sin(phi)-np.sin(psi)*np.cos(phi), np.sin(psi)*np.sin(theta)*np.sin(phi)+np.cos(psi)*np.cos(phi), np.sin(psi)*np.sin(theta)*np.cos(phi)-np.cos(psi)*np.sin(phi)], \
                            [-np.sin(theta), np.cos(theta)*np.sin(phi), np.cos(theta)*np.cos(phi)] ])

# Transpose of the Direct Cosine Matrix
    def transpose_DCM(theta,phi,psi):
        return np.transpose(DCM_transfrom(theta,phi,psi))


    def bodyAngularRatesToEulerAngler(self):
        return np.matrix( [ [1, np.sin(self.phi)*np.sin(self.theta), np.cos(self.phi)*np.tan(self.theta)], \
                    [0, np.cos(self.phi), -1*np.sin(self.phi)], \
                    [0, np.sin(self.phi)/np.cos(self.theta), np.cos(self.phi)/np.cos(self.theta)] ] )*np.matrix([ [self.P],[self.Q],[self.R]])





# Returns the dynamic drag in Newton
# p: density of the fluid
# v: linear velocity of the object to the fluid
# Cd: Drag Coefficient
# A: Reference Area
    def aerodrag(p, v, Cd, A):
        return 0.5*p*np.square(v)*Cd*A

# Model for the wind disturbances
    def wind():
        pass

# Returns the components of (m*g)
    def gravity(self):
        return np.matrix([ [-1*np.sin(self.theta)],[np.cos(self.theta)*np.sin(self.phi)],[np.cos(self.theta)*np.cos(self.phi)]])*self.m*9.81


# Acceleration in the X direction
    def update_Udot(self):
        self.Udot = (np.divide(self.X, self.m) + self.V*self.R - self.W*self.Q)

# Acceleration in the Y direction
    def update_Vdot(self):
        self.Vdot = np.divide(self.Y, self.m) - self.U*self.R + self.W*self.P

# Acceleration in the Z direction
    def update_Wdot(self):
        self.Wdot = np.divide(self.Z, self.m) + self.U*self.Q - self.V*self.P

    def update_Pdot(self):
        self.Pdot = np.divide(self.L, self.Ixx) - np.divide( (self.Izz - self.Iyy), self.Ixx)*self.Q*self.R

    def update_Qdot(self):
        self.Qdot = np.divide(self.M,self.Izz) - np.divide((self.Ixx - self.Izz), self.Ixx)*self.Q*self.R

    def update_Rdot(self):
        self.Rdot = np.divide(self.N, self.Izz) - np.divide( (self.Iyy - self.Ixx ), self.Izz)*self.P*self.Q


# Rotor lag dynamics
    def dThrust_dt(self,T,tau,Tr):
        dThrust_dt = np.divide(T,tau) + np.divide(Tr,tau)
        return dThrust_dt



    def sumForces_X(self):
        self.X = np.asscalar(self.gravity()[0])

    def sumForces_Y(self):
        self.Y = np.asscalar(self.gravity()[1])

    def sumForces_Z(self):
        self.Z = -1*(self.T1 + self.T2 + self.T3 + self.T4) + np.asscalar(self.gravity()[2])

    def sumMoment_l(self):
        self.L = self.d*(self.T4-self.T2)

    def sumMoment_m(self):
        self.M = self.d*(self.T1-self.T3)

    def sumMoment_n(self):
        self.N = self.d*(-self.T1+self.T2-self.T3+self.T4)


    def thrust(self):
        pass

    def integration(self,x):
        from scipy.integrate import quad
        f = lambda t: x
        return quad(f,self.t, self.t+self.dt)


# Determine the next state conditions
    def step(self):


        self.sumForces_X()
        self.sumForces_X()
        self.sumForces_Y()
        self.sumForces_Z()
        self.sumMoment_l()
        self.sumMoment_m()
        self.sumMoment_n()




        self.update_Udot()
        self.update_Vdot()
        self.update_Wdot()
        self.update_Pdot()
        self.update_Qdot()
        self.update_Rdot()





        self.U += self.integration(self.Udot)[0]
        self.V += self.integration(self.Vdot)[0]
        self.W += self.integration(self.Wdot)[0]


        self.P += self.integration(self.Pdot)[0]
        self.Q += self.integration(self.Qdot)[0]
        self.R += self.integration(self.Rdot)[0]


        self.theta_dot += np.asscalar(self.bodyAngularRatesToEulerAngler()[0])
        self.phi_dot += np.asscalar(self.bodyAngularRatesToEulerAngler()[1])
        self.psi_dot += np.asscalar(self.bodyAngularRatesToEulerAngler()[2])


        self.xPos += self.integration(self.U)[0]
        self.yPos += self.integration(self.V)[0]
        self.zPos += self.integration(self.W)[0]

        self.theta += self.integration(self.theta_dot)[0]
        self.phi += self.integration(self.phi_dot)[0]
        self.psi += self.integration(self.psi)[0]

        self.t += self.dt


        # Check


    def getStates(self):
        return np.r_[ self.xPos, self.yPos, self.zPos, self.theta_dot, self.phi_dot, self.psi_dot, self.theta, self.phi, self.psi]
# System Constants used from Moller

systemConstants = np.array([1,1,1,1])   # [mass,Ixx,Iyy,Izz]
initialConditions = np.array([0,0,0,0,0,0])   # [x,y,z,theta,phi,psi]


drone1 = drone(systemConstants,initialConditions)

states = np.zeros(9)

for k in range(100):
    drone1.step()
    states = np.vstack( (states, drone1.getStates() ) )



plt.figure(1)
plt.plot(states[:,0:3],'o-', mew=1, ms=8,mec='w')
plt.legend(['x','y','z', 'u','v','w', 'p', 'q', 'r'])
plt.grid()


plt.figure(2)
plt.plot(states[:,3:6],'o-', mew=1, ms=8,mec='w')
plt.legend([r'$\dot \theta$','$\dot \phi$','$\dot \psi$'])
plt.grid()


plt.figure(3)
plt.plot(states[:,6:9],'o-', mew=1, ms=8,mec='w')
plt.legend([r'$\theta$','$\phi$','$\psi$'])
plt.grid()



plt.show()
