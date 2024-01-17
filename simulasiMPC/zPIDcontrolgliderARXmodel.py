import numpy as np
from gekko import GEKKO
import matplotlib.pyplot as plt
import pickle
from PID import PID

# creating new object controller
depthPID = PID(100)
pitchPID = PID()

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
yc, uc = m.arx(p)
m.options.IMODE = 1
m.solve(disp=False)

# variable initialize
numstep = 200
yc[0].value[0] = 0
yc[1].value[0] = 0
depth_sim = np.array([0])
pitch_angle_sim = np.array([0])
depth_rate_sim = np.array([0])
setpointdepth_rate_sim = np.array([0])
BE = np.array([0])
MM = np.array([0])
depthRateNow = 0
depthNow = 0
pitchNow = 0
BE_input = BE[0]
m.time = np.linspace(0, 9, 10)

# setPoint variable
desired_depth_setpoint = 10*1000
desired_pitch_setpoint = -20
plt.figure(figsize=(6, 10))

# simulation loop
for i in range(numstep):
    # if i == 100:
    #     desired_depth_setpoint = 0
    #     desired_pitch_setpoint = 30

    # getting the value of BE and solve the model
    uc[0].value = BE
    uc[1].value = MM
    m.options.TIME_SHIFT = 0
    m.options.IMODE = 5
    m.solve(disp=False)

    # updating the time
    m.time = np.linspace(0, i+1, i+2)

    # Get the calculated input values and store to desired variable
    depthRateNow = yc[0].value[i]
    depthNow += depthRateNow
    pitchNow = yc[1].value[i]
    depth_sim = np.append(depth_sim, depthNow)
    pitch_angle_sim = np.append(pitch_angle_sim, pitchNow)
    depth_rate_sim = np.append(depth_rate_sim, depthRateNow)

    # calculating PID control output
    errorDepth = desired_depth_setpoint - depthNow
    print('ErrorDepth : ', errorDepth)
    veloDes_m = depthPID.calculate_desVelo(errorDepth)
    # convert to milimeter
    veloDes = veloDes_m
    print("VeloDes: ", veloDes)
    BE_rate = depthPID.calculatePID(veloDes, depthRateNow,
                                    2, 0, 0, lb=-17, ub=17)
    setpointdepth_rate_sim = np.append(setpointdepth_rate_sim, veloDes)
    print("BE_rate: ", BE_rate)
    BE_input = BE_rate
    MM_val = pitchPID.calculatePID(desired_pitch_setpoint, pitchNow,
                                   1.5, 0, 0, lb=-10, ub=10, limitlowval=False)
    print("MM_val: ", MM_val)

    # storing input value for visualization
    BE = np.append(BE, BE_input)
    MM = np.append(MM, MM_val)
    print(BE)
    scaledMM = MM*1

    # print(uc[0].value)
    # print(uc[1].value)
    # print(yc[0].value)
    # print(yc[1].value)

    # print(depth_sim)
    # print(pitch_angle_sim)

    # Plot the simulation results
    plt.clf()

    plt.subplot(4, 1, 1)
    plt.title("PID Control Simulation 0.1m/s Depth Rate")
    plt.plot(np.arange(i+2), scaledMM, label="Moving Mass")
    plt.plot(np.arange(i+2), BE, label="Buoyancy Engine")
    plt.ylabel('Output Value')
    plt.legend()

    plt.subplot(4, 1, 2)
    plt.plot(np.arange(i+2), depth_sim, label="Depth")
    plt.plot(np.arange(i+2), np.full(i+2, desired_depth_setpoint), color="r", linestyle='--',
             label="Setpoint Depth")
    plt.ylabel('Depth (m)')
    plt.legend()

    plt.subplot(4, 1, 3)
    plt.plot(np.arange(i+2), pitch_angle_sim, label="Pitch")
    plt.plot(np.arange(i+2), np.full(i+2, desired_pitch_setpoint), color="r", linestyle='--',
             label="Setpoint Pitch")
    plt.ylabel('Pitch degree')
    plt.legend()

    plt.subplot(4, 1, 4)  # Add a new subplot for depth rate
    plt.plot(np.arange(i+2), depth_rate_sim, label='Depth Rate')
    plt.plot(np.arange(i+2), setpointdepth_rate_sim, color='r',
             linestyle='--', label='Setpoint Depth Rate')
    plt.xlabel('Time (s)')
    plt.ylabel('Depth Rate')
    plt.legend()

    plt.draw()
    plt.pause(0.05)

plt.savefig('PIDwoDisturbance.png', dpi=300)
plt.show()

print(BE)

file_path = "simulation_data_PIDnoDisturbance.npz"

np.savez(file_path, depth_sim=depth_sim,
         depth_rate_sim = depth_rate_sim,
         setpointdepth_rate_sim = setpointdepth_rate_sim,
         pitch_angle_sim=pitch_angle_sim, BE_input=BE, MM_input=MM)

totalInputChange = np.sum(np.abs(np.diff(BE)))
print(totalInputChange)
