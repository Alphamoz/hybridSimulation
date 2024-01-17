from gekko import GEKKO
import numpy as np
import matplotlib.pyplot as plt
import pickle

p = {'a': np.array([[1.79829171,  1.91523191],  # input
                    [-0.80088764, -0.94584446]]),
     'b': np.array([[[0.00019156, 1.72345e-10]],
                   [[0.00092165, 0.04483734]]]),
     'c': np.array([0.000, -0.00])}

# Create GEKKO model
m = GEKKO(remote=False)
y = m.Array(m.CV,2)
u = m.Array(m.MV,2)
m.arx(p,y,u)

depth = y[0]
movingmass = y[1]

# rename MVs
BE = u[0]
moving_mass = u[1]

# steady state initialization
m.options.IMODE = 1
m.solve(disp=False)

# set up MPC
m.options.IMODE   = 6 # MPC
m.options.CV_TYPE = 1 # Objective type
m.options.NODES   = 2 # Collocation nodes
m.options.SOLVER  = 3 # IPOPT
m.time=np.linspace(0,120,61)

# Manipulated variables
BE.STATUS = 1  # manipulated
BE.FSTATUS = 0 # not measured
BE.DMAX = 15
BE.DCOST = 0.1
BE.UPPER = 350
BE.LOWER = -350

moving_mass.STATUS = 1  # manipulated
moving_mass.FSTATUS = 0 # not measured
moving_mass.DMAX = 1/3
moving_mass.DCOST = 0.1
moving_mass.UPPER = 50.0
moving_mass.LOWER = -50.0

# Controlled variables
depth.STATUS = 1     # drive to set point
depth.FSTATUS = 1    # receive measurement
TC1.TAU = 20       # response speed (time constant)
TC1.TR_INIT = 2    # reference trajectory
TC1.TR_OPEN = 0

TC2.STATUS = 1     # drive to set point
TC2.FSTATUS = 1    # receive measurement
TC2.TAU = 20        # response speed (time constant)
TC2.TR_INIT = 2    # dead-band
TC2.TR_OPEN = 1


# Moving Mass input variable
moving_mass = m.MV(value=0, lb=-50, ub=50)
moving_mass.STATUS = 1  # Enable control of moving mass variable
moving_mass.DMAX = 1/3  # Maximum increment in 3 seconds

# USING BE constraint
BE = m.MV(value=0, lb=-350, ub=350)
BE.STATUS = 1  # Enable control
BE.DMAX = 15
BE.DCOST = 2

# Define the output variables to be controlled
depth = m.CV(value=0)
depth_rate = m.Var(value=0)
depth.STATUS = 1  # Control objective
pitch_angle = m.CV(value=0)
pitch_angle.STATUS = 1  # Control objective


m.Equation(depth.dt() == depth_rate)

# Set the setpoints for the control objectives
pitch_angle.SP = -30  # Setpoint for pitch angle
depth.SP = 10
depth.SPHI = 10 + 0.1
depth.SPLO = 10 - 0.1
# Define the dynamic equations
yc,uc=m.arx(p, y=[depth_rate, pitch_angle], u=[BE, moving_mass])

# steady state initialization
m.options.IMODE = 1
m.solve()

# Set up the simulation
num_steps = 200  # Number of simulation steps
control_horizon = 50
time = np.linspace(0, num_steps, (num_steps + 1))
# Define the time horizon for prediction and control
m.time = np.linspace(0, control_horizon, (control_horizon + 1))
opt_moving_mass = np.zeros(num_steps)
opt_buoyancy_engine_rate = np.zeros(num_steps)
depth_sim = np.zeros(num_steps)
pitch_angle_sim = np.zeros(num_steps)
BE_input = np.zeros(num_steps)
MM_input = np.zeros(num_steps)
print(BE_input.size)
totalInputChangeListBE = np.zeros(num_steps)
totalInputChangeListMM = np.zeros(num_steps)

# Set up the MPC controller and optimization
m.options.IMODE = 6  # MPC mode control
m.options.CV_TYPE = 2  # Squared error
m.options.NODES   = 2 # Collocation nodes
m.options.SOLVER  = 3 # IPOPT
m.time=np.linspace(0,120,61)

print(yc[0].VALUE) #output depth dan pitch sistem
print(moving_mass.NEWVAL)
print(BE.NEWVAL)
print(depth.VALUE)


for i in range(1,num_steps-1):
    # Update measured values of outputs
    depth.MEAS = yc[0].VALUE[i-1]
    pitch_angle.MEAS = yc[1].VALUE[i-1]
    
    # Update setpoints of CVs (optional)
    # You can change the setpoints dynamically here if you want
    # For example, you can use a sinusoidal function or a step function
    # depth.SP = 10 + 5*np.sin(0.1*i)
    # pitch_angle.SP = -30 + 10*np.heaviside(i-100,0)
    
    # Update time horizon for prediction and control
    # m.time = np.linspace(i,i+control_horizon,(control_horizon+1))
    
    # Solve the optimization problem
    m.solve()
    
    # Store optimal values of MVs and CVs
    opt_moving_mass[i+1] = moving_mass.NEWVAL
    opt_buoyancy_engine_rate[i+1] = BE.NEWVAL
    moving_mass.MEAS = opt_moving_mass[i+1]
    BE.MEAS = opt_buoyancy_engine_rate[i+1]
    yc[0].VALUE 
    # depth_sim[i+1] = depth.MEAS
    # pitch_angle_sim[i+1] = pitch_angle.MEAS
    print(yc[0].VALUE) #output depth dan pitch sistem
    print(moving_mass.NEWVAL)
    print(BE.NEWVAL)
    print(depth.VALUE)

    plt.clf()
    # plt.figure()
    plt.subplot(4, 1, 1)
    plt.plot(time[0:i+1], depth_sim[0:i+1], label='Depth')
    plt.axhline(y=depth.SP, color='r', linestyle='--', label='Setpoint')
    plt.xlabel('Time (s)')
    plt.ylabel('Depth')
    plt.legend()

    plt.subplot(4, 1, 2)
    plt.plot(time[0:i+2], pitch_angle_sim[0:i+2], label='Pitch Angle')
    plt.axhline(y=pitch_angle.SP, color='r', linestyle='--', label='Setpoint')
    plt.xlabel('Time (s)')
    plt.ylabel('Pitch Angle')
    plt.legend()

    plt.subplot(4, 1, 3)
    plt.step(time[0:i], opt_moving_mass[0:i], label='Moving Mass')
    plt.step(time[0:i], opt_buoyancy_engine_rate[0:i], label='Buoyancy Engine')
    plt.xlabel('Time (s)')
    plt.ylabel('Input')
    plt.legend()

    # plt.subplot(4, 1, 4)
    # plt.step(time[0:i], totalInputChangeListBE[0:i], '--', label='Moving Mass')
    # plt.step(time[0:i], totalInputChangeListMM[0:i], '--', label='Moving Mass')
    # plt.xlabel('Time (s)')
    # plt.ylabel('Input')
    # plt.legend()

    plt.draw()
    plt.pause(0.05)


# # # Plot the results
# # plt.show()

# # totalInputChange = np.sum(np.abs(np.diff(BE_input)))

# # print(totalInputChange)
