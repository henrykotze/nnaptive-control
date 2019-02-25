
    systemConstants = np.array([1,1,1,1])   # [mass,Ixx,Iyy,Izz]
    initialConditions = np.array([0,0,0,0,0,0])   # [x,y,z,theta,phi,psi]


    drone1 = drone(systemConstants,initialConditions)

    states = np.zeros(9)
    save_data = np.zeros(2)

    for k in range(3):
        drone1.step()

        print(drone1.getStates())
        states = np.vstack( (states, drone1.getStates() ) )


    plt.rc('text', usetex=True)
    plt.rc('font', family='serif')
    plt.rc('font', size=12)



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
