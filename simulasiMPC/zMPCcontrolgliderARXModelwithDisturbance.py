from gekko import GEKKO
import numpy as np
import matplotlib.pyplot as plt
import pickle
from PID.PID import PID

# with open('p_values.pickle', 'rb') as file:
#     p = pickle.load(file)

# print(p)



p = {'a': np.array([[1.94952658,  1.90882498],  # input
                    [-0.94964712, -0.94044732]]),
     'b': np.array([[[3.6075e-6, 1.72345e-11]],
                   [[0.00056117, 0.04801735]]]),
     'c': np.array([0.000, -0.00])}
# Create GEKKO model
m = GEKKO(remote=False)

# Moving Mass input variable
moving_mass = m.MV(value=0, lb=-50, ub=50)
moving_mass.STATUS = 1  # Enable control of moving mass variable
moving_mass.DMAX = 1/3  # Maximum increment in 3 seconds

# USING BE constraint
BE = m.MV(value=0, lb=-350, ub=350)
BE.STATUS = 1   # Enable control
BE.DMAX = 15    # Input change constraint
BE.DCOST = 1    # Punish for fluctuative change

# Define the output variables to be controlled
# setPointDepth = 10
# setPointPitch = -30
depth = m.CV(value=0)
depth_rate = m.Var(value=0)
# depth_rate.UPPER=0.5
# depth.STATUS = 1  # Control objective

m.Equation(depth_rate == depth.dt())
m.Equation(depth > 0)

pitch_angle = m.CV(value=0)
# pitch_angle.STATUS = 1  # Control objectiveq
pitch_angle.SP = -20  # Setpoint for pitch angle
# pitch_angle.TAU=35
depth.SP = 20

setpointdepthrate = 0.1
depthrateinit = setpointdepthrate
veloCalculation = PID(setpointdepthrate)

m.Minimize(100000*((depth_rate - setpointdepthrate))**2)
m.Minimize(1*((pitch_angle - pitch_angle.SP))**2)

m.arx(p, y=[depth_rate, pitch_angle], u=[BE, moving_mass])

# Set up the simulation
num_steps = 700  # Number of simulation steps
control_horizon = 15
time = np.linspace(0, num_steps, (num_steps + 1))
# Define the time horizon for prediction and control
m.time = np.arange(0, control_horizon*1+1, 1)
opt_moving_mass = np.zeros(num_steps)
opt_buoyancy_engine_rate = np.zeros(num_steps)
depth_sim = np.zeros(num_steps)
depth_rate_sim = np.zeros(num_steps)
setpointdepth_rate_sim = np.zeros(num_steps)
pitch_angle_sim = np.zeros(num_steps)
BE_input = np.zeros(num_steps)
MM_input = np.zeros(num_steps)
print(BE_input.size)
totalInputChangeListBE = np.zeros(num_steps)
totalInputChangeListMM = np.zeros(num_steps)
plt.figure(figsize=(6,8))

# Set up the MPC controller and optimization
m.options.IMODE = 6  # MPC mode control
# m.options.CV_TYPE = 2  # Squared error
m.options.MAX_ITER = 300  # Maximum number of iterations
m.options.SOLVER= 3 #IPOPT
m.options.CTRL_HOR = 3
m.options.CTRL_TIME = 1
m.options.PRED_HOR = 40
m.options.PRED_TIME = 1
# m.options.MV_TYPE = 1  # Use dynamic control horizon
solvingtime = []

noise_std_dev = 0.01

for i in range(1,num_steps-1):
    try:
        m.solve(disp=True, debug=1)
        solvingtime.append(m.options.SOLVETIME)
    except:
        print('Solution not found')
    # Get the optimized inputs
    disturbance = np.random.normal(loc=0, scale=0.1)
    opt_moving_mass[i] = moving_mass.NEWVAL
    opt_buoyancy_engine_rate[i] = BE.NEWVAL

    # Simulate the model and store the outputs
    depth_sim[i] = depth.VALUE[0]
    pitch_angle_sim[i] = pitch_angle.VALUE[0]
    depth_rate_sim[i] = depth_sim[i]-depth_sim[i-1]
    setpointdepth_rate_sim[i] = setpointdepthrate

    # Apply inputs to the model
    moving_mass.MEAS = opt_moving_mass[i]
    BE.MEAS = opt_buoyancy_engine_rate[i]
    BE_input[i] = BE.NEWVAL
    MM_input[i] = moving_mass.NEWVAL
    opt_moving_mass[i] *= 20



    depth_sim[i] += np.random.normal(loc=0, scale=noise_std_dev)
    pitch_angle_sim[i] += np.random.normal(loc=0, scale=0.1)

    # updating objective function each loop
    
    veloDes = veloCalculation.calculate_desVelo(depth.SP-depth_sim[i])
    setpointdepthrate = veloDes
    m._objectives.clear()
    m.Minimize(100000*((depth_rate - setpointdepthrate))**2)
    m.Minimize(1*((pitch_angle - pitch_angle.SP))**2)

    plt.clf()
    
    plt.subplot(4, 1, 1)
    plt.title("MPC Control Simulation with Disturbance")
    plt.plot(time[0:i], opt_moving_mass[0:i], label='Moving Mass')
    plt.plot(time[0:i], opt_buoyancy_engine_rate[0:i], label='Buoyancy Engine')
    # plt.xlabel('Time (s)')
    plt.ylabel('Input')
    plt.legend()
    

    plt.subplot(4, 1, 2)
    plt.plot(time[0:i], depth_sim[0:i], label='Depth')
    plt.axhline(y=depth.SP, color='r', linestyle='--', label='Setpoint Depth')
    # plt.xlabel('Time (s)')
    plt.ylabel('Depth')
    plt.legend()
    

    plt.subplot(4, 1, 3)
    plt.plot(time[0:i], pitch_angle_sim[0:i], label='Pitch Angle')
    plt.axhline(y=pitch_angle.SP, color='r', linestyle='--', label='Setpoint Depth')
    # plt.xlabel('Time (s)')
    plt.ylabel('Pitch Angle')
    plt.legend()

    plt.subplot(4, 1, 4)  # Add a new subplot for depth rate
    plt.plot(time[0:i], depth_rate_sim[0:i], label='Depth Rate')
    plt.plot(time[0:i],setpointdepth_rate_sim[0:i], color='r', linestyle='--', label='Setpoint Depth Rate')
    plt.xlabel('Time (s)')
    plt.ylabel('Depth Rate')
    plt.legend()
   

    plt.draw()
    plt.pause(0.05)


# Plot the results
plt.savefig('MPCwithDisturbance.png', dpi=300)
plt.show()

file_path = "simulation_data_MPCwithDisturbance.npz"

np.savez(file_path, depth_sim=depth_sim, pitch_angle_sim=pitch_angle_sim, BE_input=BE_input, MM_input=opt_moving_mass)

# totalInputChange = 0
# for i in range(BE_input.size-1):
#     totalInputChange += abs(BE_input[i+1]-BE_input[i])

# print(totalInputChange)
