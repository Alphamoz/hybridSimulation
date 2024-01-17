from gekko import GEKKO
import numpy as np
import matplotlib.pyplot as plt
import pickle
from PID import PID
import time as t

m = GEKKO(remote=False)
#       dynamics dr, dynamics pitch
# a->   [a1(k-1), a2(k-1)]
#       [a1(k-2),a2(k-2)]   etc...

# dynamics input BE, dynamics input MM for dr
# dynamics input BE, dynamics input MM for pitch
# b -> b for BE[[b1(k-1), b2(k-1)], 
#               [b1(k-2), b2(k-2)]
#   -> b for MM[[b1(k-1), b2(k-1)], [b1(k-2), b2(k-2)] 
with open('p_br_pitch.pickle', 'rb') as file:
    p_pitch = pickle.load(file)
    
with open('p_br_dr.pickle', 'rb') as file:
    p_dr = pickle.load(file)

print(p_pitch)
print(p_dr)

p = {'a': np.array([[ p_dr['a'][0][0] , p_pitch['a'][0][0]]
       ]), 'b': np.array([[[ p_dr['b'][0][0][0], 0],
                           [p_dr['b'][0][1][0],0]],
                          [[ p_pitch['b'][0][0][0],p_pitch['b'][0][0][1]],
                           [0,0]]
       ]), 'c': np.array([0,0])}

# Create GEKKO model
m = GEKKO(remote=False)

# Moving Mass input variable
MM = m.Var(value=0, lb=-250, ub = 250)
mr = m.MV(value=0, lb=-10, ub=10)
mr.STATUS=1

# USING BE constraint
BE = m.Var(value=0, lb=-350, ub = 350)
# BE.value = 350
# print("BE NYA BOS!",BE.value)
# t.sleep(5)
# BE.STATUS = 1  # Enable control
# BE.DMAXHI = 15
# BE.DMAXLO = -15
# BE.DCOST = 1
br = m.MV(value=0, lb=-15, ub=15)
br.STATUS=1

# Define the output variables to be controlled
# setPointDepth = 10
# setPointPitch = -30
depth = m.Var(value=0)
depth_rate = m.CV(value=0)
depth_rate.FSTATUS=1
# depth_rate.LOWER=-0.1
# depth_rate.UPPER=0.1
# depth_rate.STATUS = 1  # Control objective

m.Equation(depth == m.integral(depth_rate))
m.Equation(BE == m.integral(br))
m.Equation(MM == m.integral(mr))

m.Equation(depth > 0)

pitch_angle = m.CV(value=0)
pitch_angle.FSTATUS = 1  # Control objective

# Set the setpoints for the control objectives
pitch_angle.SP = -20  # Setpoint for pitch angle
# pitch_angle.TAU=35
depthsetpoint = 20000
# depth.TAU = 50
# depth.SPHI = depth.SP + 0.1
# depth.SPLO = depth.SP - 0.1
# pitch_angle.SPHI = pitch_angle.SP + 1
# pitch_angle.SPLO = pitch_angle.SP - 1

setpointdepthrate = 100
depthrateinit = setpointdepthrate
veloCalculation = PID(setpointdepthrate)

m.Minimize(1*((depth_rate - setpointdepthrate))**2)
# m.Minimize(1*((depth - depthsetpoint))**2)
m.Minimize(1*((pitch_angle - pitch_angle.SP))**2)
m.Minimize(0.0001*((br)**2)+ 0.0001*((mr)**2))

# Define the dynamic equations
m.arx(p, y=[depth_rate, pitch_angle], u=[br, mr])


# Set up the simulation
num_steps = 300  # Number of simulation steps
control_horizon = 15
prediction_horizon = 15
time = np.linspace(0, num_steps-1, (num_steps))
# Define the time horizon for prediction and control
m.time = np.arange(0, control_horizon+1, 1)

# print("BE NYA BOS!",BE.value)
# t.sleep(2)

opt_moving_mass = np.zeros(num_steps)
opt_buoyancy_engine_rate = np.zeros(num_steps)
depth_sim = np.zeros(num_steps)
depth_rate_sim = np.zeros(num_steps)
setpointdepth_rate_sim = np.zeros(num_steps)
setpointdepth_sim = np.zeros(num_steps)
setpointpitch_angle_sim = np.zeros(num_steps)
pitch_angle_sim = np.zeros(num_steps)
BE_input = np.zeros(num_steps)
MM_input = np.zeros(num_steps)
# print(BE_input.size)
totalInputChangeListBE = np.zeros(num_steps)
totalInputChangeListMM = np.zeros(num_steps)
plt.figure(figsize=(6, 8))

# Set up the MPC controller and optimization
m.options.IMODE = 6  # MPC mode control
# m.options.CV_TYPE = 2  # Squared error
m.options.MAX_ITER = 100  # Maximum number of iterations
m.options.SOLVER = 1  # IPOPT
m.options.CTRL_HOR = 3
m.options.CTRL_TIME = 1
m.options.PRED_HOR = prediction_horizon
m.options.PRED_TIME = 1
# m.options.MV_TYPE = 1  # Use dynamic control horizon
solvingtime = []
lastsin=0

# asumsi, inputnya sudah dapat yang optimized, outputnya juga sudah tau.
for i in range(1, num_steps):
    # Read depth rate data
    print("Ini nilai depth rate yang seharusnya dari ARX",depth_rate.VALUE)
    print("Ini nilai input BE yang seharusnya dari ARX",br.NEWVAL)
    # depth_rate.MEAS = 20
    # t.sleep(2)
    try:
        m.solve(disp=False)
        solvingtime.append(m.options.SOLVETIME)
    except:
        print('Solution not found')
    print("Ini nilai depth rate yang seharusnya dari ARX setelah diupdate",depth_rate.VALUE)
    sinusoidal = 10 * np.sin(1/10 * np.pi * i)
    depth_rate.MEAS = depth_rate.VALUE[1] + sinusoidal-lastsin
    pitch_angle.MEAS = pitch_angle.VALUE[1] + sinusoidal/2-lastsin/2
    # depth.value = depth_sim[i]+ depth_rate.MEAS
    lastsin = sinusoidal
    print(depth_rate.VALUE)
    opt_moving_mass[i] = mr.NEWVAL
    opt_buoyancy_engine_rate[i] = br.NEWVAL
    # Simulate the model and store the outputs
    
    depth_sim[i] = depth_sim[i-1]+ depth_rate.MEAS
    
    pitch_angle_sim[i] = pitch_angle.VALUE[1]
    depth_rate_sim[i] = depth_rate.MEAS
    setpointdepth_rate_sim[i] = setpointdepthrate
    setpointdepth_sim[i] = depthsetpoint
    setpointpitch_angle_sim[i]=pitch_angle.SP
    
    
    # Apply inputs to the model
    mr.MEAS = mr.NEWVAL
    br.MEAS = br.NEWVAL
    print("BE NYA BOS!",BE.value)

    BE_input[i] = br.NEWVAL
    MM_input[i] = mr.NEWVAL
    # opt_moving_mass[i] *= 20

    inputChangeListBE = np.abs(np.diff(opt_buoyancy_engine_rate))
    totalInputChangeBE = np.sum(inputChangeListBE)

    # updating objective function each loop
    veloDes = veloCalculation.calculate_desVelo(depthsetpoint-depth_sim[i])
    setpointdepthrate = veloDes
    m._objectives.clear()
    R1=0.0001
    R2=0.0001
    weight_varying = False
    if(weight_varying):
        if(abs(setpointdepthrate-depth_rate.MEAS)<30):
            R1=1000
        if(abs(pitch_angle.SP-pitch_angle.VALUE[1])<6):
            R2=1000
    m.Minimize(10*((depth_rate - setpointdepthrate))**2)
    m.Minimize(10*((pitch_angle - pitch_angle.SP))**2)
    m.Minimize(R1*((br)**2)+ R2*((mr)**2))

    plt.clf()
    # plt.figsize()
    plt.subplot(4, 1, 1)
    plt.title("MPC Control Gliding Simulation with Disturbance and Fixed Weight".format())
    plt.plot(time[0:i], opt_moving_mass[0:i],
             label='Moving Mass')
    plt.plot(time[0:i], opt_buoyancy_engine_rate[0:i], label='Buoyancy Engine')
    plt.ylabel('Input')
    plt.legend()

    plt.subplot(4, 1, 2)
    plt.plot(time[0:i], depth_sim[0:i], label='Depth (mm)')
    plt.plot(time[0:i],setpointdepth_sim[0:i], color='r', linestyle='--', label='Setpoint Depth')
    plt.ylabel('Depth')
    plt.legend()

    plt.subplot(4, 1, 3)
    plt.plot(time[0:i], pitch_angle_sim[0:i], label='Pitch Angle')
    plt.plot(time[0:i],setpointpitch_angle_sim[0:i], color='r',
                linestyle='--', label='Setpoint Pitch')
    # plt.xlabel('Time (s)')
    plt.ylabel('Pitch Angle')
    plt.legend()

    plt.subplot(4, 1, 4)  # Add a new subplot for depth rate
    plt.plot(time[0:i], depth_rate_sim[0:i], label='Depth Rate (mm/s)')
    plt.plot(time[0:i], setpointdepth_rate_sim[0:i], color='r',
             linestyle='--', label='Setpoint Depth Rate (mm/s)')
    plt.xlabel('Time (s)')
    plt.ylabel('Depth Rate')
    plt.legend()

    plt.draw()
    # plt.pause(0.05)


# Plot the results
# plt.savefig('MPCwoDisturbance{}{}.png'.format(
#     depthrateinit, control_horizon), dpi=300)
plt.show()

# calculating input changes
totalInputChange = np.sum(np.abs(np.diff(BE_input)))

print(totalInputChange)

file_path = "simulation_data_MPCnoDisturbanceweightvarying{}.npz".format(weight_varying)

np.savez(file_path, depth_sim=depth_sim, pitch_angle_sim=pitch_angle_sim,
         depth_rate_sim=depth_rate_sim, BE_input=BE_input, MM_input=MM_input,
         setpointdepth_rate_sim=setpointdepth_rate_sim,
         setpointpitch_angle_sim=pitch_angle.SP, setpointdepth_sim=depthsetpoint, solvingtime=solvingtime)

# file_path = "simulation_data_MPCnoDisturbance{}{}.npz".format(depthrateinit,control_horizon)
print(np.mean(np.array(solvingtime)))
# settling_start_indexDepth = np.argmax(np.abs(depth_sim - depth.SP) <= depth.SP * 0.05)
# settling_start_indexPitch = np.argmax(np.abs(pitch_angle_sim - pitch_angle.SP) <= pitch_angle.SP * 0.05)

# if settling_start_indexDepth > 0:
#     settling_time = time[settling_start_indexDepth]
# else:
#     # If the settling threshold is not reached, consider settling time as NaN
#     settling_time = np.nan

# print("Settling Time Depth to 5%:", settling_time, "seconds")

# if settling_start_indexPitch > 0:
#     settling_time = time[settling_start_indexPitch]
# else:
#     # If the settling threshold is not reached, consider settling time as NaN
#     settling_time = np.nan

# print("Settling Time to 5%:", settling_time, "seconds")
