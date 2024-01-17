import numpy as np



num_steps = 300

time = np.linspace(0, num_steps, num_steps + 1)

def calculate_settling_time(data, setpoint, percentage):
    settling_start_index = np.argmax(np.abs(data - setpoint) <= abs(setpoint*percentage))
    print(settling_start_index)
    if settling_start_index > 0:
        settling_time = time[settling_start_index]
    else:
        settling_time = np.nan
    return settling_time

def calculate_rise_time(data, setpoint, threshold):
    rise_start_index = np.argmax(np.abs(data) >= abs(threshold*setpoint))
    # print(rise_start_index)
    if rise_start_index > 0:
        rise_time = time[rise_start_index]
    else:
        rise_time = np.nan
    return rise_time

def calculateEnergy(data):
    totalInputChange = np.sum(np.abs(np.diff(data)))
    return totalInputChange

def calculate_overshoot(data, setpoint):
    max_value = np.max(np.abs(data))
    overshoot = max_value - abs(setpoint)
    overshoot_percentage = (overshoot / abs(setpoint)) * 100
    return overshoot_percentage
# for i in [0.1,0.2,0.3,0.4]:
#     MPC_no_Disturbance_Data = np.load('simulation_data_MPCnoDisturbance{}.npz'.format(i))

#     loaded_depth_sim = MPC_no_Disturbance_Data['depth_sim']
#     loaded_pitch_angle_sim = MPC_no_Disturbance_Data['pitch_angle_sim']
#     loaded_BE = MPC_no_Disturbance_Data['BE_input']
#     loaded_MM = MPC_no_Disturbance_Data['MM_input']
#     loaded_depth_rate_sim = MPC_no_Disturbance_Data['depth_rate_sim']
#     # print(loaded_BE)

#     # print(loaded_depth_sim)
#     # print(loaded_pitch_angle_sim)

#     settling_time_depth = calculate_settling_time(loaded_depth_sim, 20, 0.05)
#     rise_time_depth = calculate_rise_time(loaded_depth_sim, 20, 0.9)
#     energy = calculateEnergy(loaded_BE)
#     os = calculate_overshoot(loaded_depth_sim,20)
#     print(settling_time_depth, rise_time_depth, energy, os)

#     settling_time_pitch = calculate_settling_time(loaded_pitch_angle_sim, -20, 0.05)
#     rise_time_depth = calculate_rise_time(loaded_pitch_angle_sim, -20, 0.9)
#     energy = calculateEnergy(loaded_MM)
#     os = calculate_overshoot(loaded_pitch_angle_sim, -20)
#     print(settling_time_pitch, rise_time_depth, energy, os)

#     settling_time_pitch = calculate_settling_time(loaded_depth_rate_sim, 0.1, 0.05)
#     rise_time_depth = calculate_rise_time(loaded_depth_rate_sim, 0.1, 0.9)
#     os = calculate_overshoot(loaded_depth_rate_sim, -i)
#     print(settling_time_pitch, rise_time_depth, energy, os)



PID_no_Disturbance_Data = np.load('simulation_data_PIDnoDisturbance.npz')

loaded_depth_sim = PID_no_Disturbance_Data['depth_sim']
loaded_pitch_angle_sim = PID_no_Disturbance_Data['pitch_angle_sim']
loaded_BE = PID_no_Disturbance_Data['BE_input']
loaded_MM = PID_no_Disturbance_Data['MM_input']



settling_time_depth = calculate_settling_time(loaded_depth_sim, 20, 0.05)
rise_time_depth = calculate_rise_time(loaded_depth_sim, 20, 0.9)
energy = calculateEnergy(loaded_BE)
os = calculate_overshoot(loaded_depth_sim,20)
print(settling_time_depth, rise_time_depth, energy, os)

settling_time_pitch = calculate_settling_time(loaded_pitch_angle_sim, -20, 0.05)
rise_time_depth = calculate_rise_time(loaded_pitch_angle_sim, -20, 0.9)
# print(loaded_MM)
energy = calculateEnergy(loaded_MM)
os = calculate_overshoot(loaded_pitch_angle_sim, -20)
print(settling_time_pitch, rise_time_depth, energy, os)



MPC_no_Disturbance_Data = np.load('simulation_data_MPCnoDisturbance0.3.npz')

loaded_depth_sim = MPC_no_Disturbance_Data['depth_sim']
loaded_pitch_angle_sim = MPC_no_Disturbance_Data['pitch_angle_sim']
loaded_BE = MPC_no_Disturbance_Data['BE_input']
loaded_MM = MPC_no_Disturbance_Data['MM_input']
loaded_depth_rate_sim = MPC_no_Disturbance_Data['depth_rate_sim']
# print(loaded_BE)

# print(loaded_depth_sim)
# print(loaded_pitch_angle_sim)

settling_time_depth = calculate_settling_time(loaded_depth_sim, 20, 0.05)
rise_time_depth = calculate_rise_time(loaded_depth_sim, 20, 0.9)
energy = calculateEnergy(loaded_BE)
os = calculate_overshoot(loaded_depth_sim, 20)
print(settling_time_depth, rise_time_depth, energy, os)

settling_time_pitch = calculate_settling_time(loaded_pitch_angle_sim, -20, 0.05)
rise_time_depth = calculate_rise_time(loaded_pitch_angle_sim, -20, 0.9)
energy = calculateEnergy(loaded_MM)
os = calculate_overshoot(loaded_pitch_angle_sim,-20)
# print(loaded_MM)
print(settling_time_pitch, rise_time_depth, energy,os)

settling_time_pitch = calculate_settling_time(loaded_depth_rate_sim, 0.1, 0.05)
rise_time_depth = calculate_rise_time(loaded_depth_rate_sim, 0.1, 0.9)
os = calculate_overshoot(loaded_depth_rate_sim, 0.1)
print(settling_time_pitch, rise_time_depth, energy, os)


# PID_no_Disturbance_Data = np.load('simulation_data_MPCnoDisturbance0.4.npz')

# loaded_depth_sim = PID_no_Disturbance_Data['depth_sim']
# loaded_pitch_angle_sim = PID_no_Disturbance_Data['pitch_angle_sim']
# loaded_BE = PID_no_Disturbance_Data['BE_input']
# loaded_MM = PID_no_Disturbance_Data['MM_input']


# settling_time_depth = calculate_settling_time(loaded_depth_sim, 20, 0.05)
# rise_time_depth = calculate_rise_time(loaded_depth_sim, 20, 0.9)
# energy = calculateEnergy(loaded_BE)
# os = calculate_overshoot(loaded_depth_sim,20)
# print(settling_time_depth, rise_time_depth, energy,os)

# settling_time_pitch = calculate_settling_time(loaded_pitch_angle_sim, -20, 0.05)
# rise_time_depth = calculate_rise_time(loaded_pitch_angle_sim, -20, 0.9)
# # print(loaded_MM)
# energy = calculateEnergy(loaded_MM)
# os = calculate_overshoot(loaded_pitch_angle_sim,-30)
# print(settling_time_pitch, rise_time_depth, energy,os)