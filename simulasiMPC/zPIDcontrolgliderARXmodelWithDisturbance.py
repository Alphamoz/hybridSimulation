import numpy as np
from gekko import GEKKO
import matplotlib.pyplot as plt
import pickle
from PID.PID import PID

# creating new object controller
depthPID = PID()
pitchPID = PID()

with open('p_values.pickle', 'rb') as file:
    p = pickle.load(file)

# initializing parameter for model
p = {'a': np.array([[1.79729171,  1.91523191],
                    [-0.80088764, -0.94584446]]),
     'b': np.array([[[0.00019156, 1.72345e-10]],
                   [[0.00042165, 0.04483734]]]),
     'c': np.array([0.000, -0.00])}

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
BE = np.array([0])
MM = np.array([0])
depthRateNow = 0
depthNow = 0
pitchNow = 0
BE_input = BE[0]
m.time = np.linspace(0, 100, 101)

# setPoint variable
desired_depth_setpoint = 10
desired_pitch_setpoint = -30
noise_std_dev = 0.1

plt.figure(figsize=(6,10))
# simulation loop
for i in range(numstep):
    # getting the value of BE and solve the model
    uc[0].value = BE
    uc[1].value = MM
    m.options.TIME_SHIFT = 0
    m.options.IMODE = 5
    m.solve(disp=False)

    # updating the time
    m.time = np.linspace(0, i+1, i+2)

    depthRateNow = yc[0].value[i]
    pitchNow = yc[1].value[i]
    depthRateNow += np.random.normal(loc=0, scale=0.05)
    pitchNow += np.random.normal(loc=0, scale=0.05)
    depthNow += depthRateNow
    depth_sim = np.append(depth_sim, depthNow)
    pitch_angle_sim = np.append(pitch_angle_sim, pitchNow)

    # calculating PID control output
    errorDepth = desired_depth_setpoint - depthNow
    print('ErrorDepth : ', errorDepth)
    veloDes = depthPID.calculate_desVelo(errorDepth)
    BE_rate = depthPID.calculatePID(veloDes, depthRateNow,
                                    10, 0, 200, lb=-15, ub=15)
    print("BE_rate: ", BE_rate)
    BE_input += BE_rate
    MM_val = pitchPID.calculatePID(desired_pitch_setpoint, pitchNow,
                                   0.03, 0, 0.0, lb=-50, ub=50)
    print("MM_val: ", MM_val)

    # apply constraint of input value
    BE_input = max(min(BE_input, 350), -350)
    if (MM_val > 1/3):
        MM_val = MM[i] + 1/3
    elif (MM_val < -1/3):
        MM_val = MM[i] - (1/3)
    else:
        MM_val += MM[i] 

    # storing input value for visualization
    BE = np.append(BE, BE_input)
    MM = np.append(MM, MM_val)
    print(BE)

    # Get the calculated input values and store to desired variable

    # depth_sim[-1] += np.random.normal(loc=0, scale=0.1)
    # pitch_angle_sim[-1] += np.random.normal(loc=0, scale=noise_std_dev)

    plt.clf()

    plt.subplot(3, 1, 1)
    plt.title("PID Control Simulation with Disturbance")
    plt.plot(np.arange(i+2), MM, label="Moving Mass")
    plt.plot(np.arange(i+2), BE, label="Buoyancy Engine")
    plt.xlabel('Time Step')
    plt.ylabel('Output Value')
    plt.legend()

    plt.subplot(3, 1, 2)
    plt.plot(np.arange(i+2), depth_sim, label="Depth")
    plt.plot(np.arange(i+2), np.full(i+2, desired_depth_setpoint), color="r", linestyle='--',
             label="Setpoint Depth")
    plt.xlabel('Time Step')
    plt.ylabel('Value')
    plt.legend()

    plt.subplot(3, 1, 3)
    plt.plot(np.arange(i+2), pitch_angle_sim, label="Pitch")
    plt.plot(np.arange(i+2), np.full(i+2, desired_pitch_setpoint), color="r", linestyle='--',
             label="Setpoint Pitch")
    plt.xlabel('Time Step')
    plt.ylabel('Value')
    plt.legend()

    plt.draw()
    plt.pause(0.05)
plt.savefig('PIDwithDisturbance.png', dpi=300)
plt.show()

file_path = "simulation_data_PIDwithDisturbance.npz"

np.savez(file_path, depth_sim=depth_sim, pitch_angle_sim=pitch_angle_sim, BE_input=BE, MM_input=MM)

print(BE)

totalInputChange = np.sum(np.abs(np.diff(BE)))
print(totalInputChange)
