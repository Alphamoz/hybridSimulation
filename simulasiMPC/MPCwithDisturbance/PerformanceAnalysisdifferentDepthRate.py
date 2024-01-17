import numpy as np



num_steps = 298

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
    totalInputChange = np.sum(np.abs(data))
    return totalInputChange

def calculate_overshoot(data, setpoint):
    max_value = np.max(np.abs(data))
    overshoot = max_value - abs(setpoint)
    overshoot_percentage = (overshoot / abs(setpoint)) * 100
    return overshoot, overshoot_percentage
def calculate_steady_error(data, setpoint,settlingpoint):
    dataAfterSettling = data[int(settlingpoint):]
    meanError = np.mean(np.abs(dataAfterSettling-setpoint))
    return meanError

for i in [False, True]:
    MPC_no_Disturbance_Data = np.load('simulation_data_MPCnoDisturbanceweightvarying{}.npz'.format(i))
    print(MPC_no_Disturbance_Data.files)
    loaded_depth_sim = MPC_no_Disturbance_Data['depth_sim']
    loaded_pitch_angle_sim = MPC_no_Disturbance_Data['pitch_angle_sim']
    loaded_BE = MPC_no_Disturbance_Data['BE_input']
    loaded_MM = MPC_no_Disturbance_Data['MM_input']
    loaded_depth_rate_sim = MPC_no_Disturbance_Data['depth_rate_sim']
    setpointdepth_rate_sim = MPC_no_Disturbance_Data["setpointdepth_rate_sim"]
    # print(loaded_BE)

    # print(loaded_depth_sim)
    # print(loaded_pitch_angle_sim)

    settling_time_depth = calculate_settling_time(loaded_depth_sim, 20000, 0.05)
    rise_time_depth = calculate_rise_time(loaded_depth_sim, 20000, 0.9)
    energy = calculateEnergy(loaded_BE)
    max_os_depth, os_depth = calculate_overshoot(loaded_depth_sim,20000)
    steady_error = calculate_steady_error(loaded_depth_sim,20000, settling_time_depth)
    # maximum = np.max(loaded_BE)
    # minimum = np.min(loaded_BE)
    std_deviation_BE = np.std(loaded_BE)
    std_deviation_MM = np.std(loaded_MM)
    # Menghitung perubahan antar nilai berurutan
    differences = np.diff(loaded_BE)

    # Menghitung frekuensi perubahan yang signifikan
    # threshold = 1  # Atur ambang sesuai kebutuhan Anda
    # significant_changes = np.where(np.abs(differences) > threshold)[0]
    
    # frequency_of_changes = len(significant_changes) / len(loaded_BE)
    print("Depth for {}".format(i), settling_time_depth, rise_time_depth, energy, os_depth, steady_error, "with max os", max_os_depth)

    settling_time_pitch = calculate_settling_time(loaded_pitch_angle_sim, -20, 0.05)
    rise_time_pitch = calculate_rise_time(loaded_pitch_angle_sim, -20, 0.9)
    energy = calculateEnergy(loaded_MM)
    max_os_pitch, os_pitch = calculate_overshoot(loaded_pitch_angle_sim, -20)
    steady_error = calculate_steady_error(loaded_pitch_angle_sim, -20, settling_time_pitch)
    print("Pitch for {}".format(i),settling_time_pitch, rise_time_pitch, energy, os_pitch, steady_error, "with max os", max_os_pitch)

    settling_time_depth_rate = calculate_settling_time(loaded_depth_rate_sim, 100, 0.05)
    rise_time_depth_rate = calculate_rise_time(loaded_depth_rate_sim, 100, 0.9)
    max_os_dr,os_dr = calculate_overshoot(loaded_depth_rate_sim, 100)
    steady_error = calculate_steady_error(loaded_depth_rate_sim, setpointdepth_rate_sim[int(settling_time_depth_rate):], settling_time_depth_rate)

    print("Depth Rate for {}".format(i),settling_time_depth_rate, rise_time_depth_rate, energy, os_dr,steady_error, "with max os", max_os_dr)
    print("Tulis gan!",rise_time_depth, rise_time_pitch, rise_time_depth_rate,os_depth,os_pitch, os_dr ,std_deviation_BE,std_deviation_MM)


# PID_no_Disturbance_Data = np.load('simulation_data_MPCnoDisturbance0.2.npz')

# loaded_depth_sim = PID_no_Disturbance_Data['depth_sim']
# loaded_pitch_angle_sim = PID_no_Disturbance_Data['pitch_angle_sim']
# loaded_BE = PID_no_Disturbance_Data['BE_input']
# loaded_MM = PID_no_Disturbance_Data['MM_input']


# settling_time_depth = calculate_settling_time(loaded_depth_sim, 20, 0.05)
# rise_time_depth = calculate_rise_time(loaded_depth_sim, 20, 0.9)
# energy = calculateEnergy(loaded_BE)
# os = calculate_overshoot(loaded_depth_sim,20)
# print(settling_time_depth, rise_time_depth, energy, os)

# settling_time_pitch = calculate_settling_time(loaded_pitch_angle_sim, -20, 0.05)
# rise_time_depth = calculate_rise_time(loaded_pitch_angle_sim, -20, 0.9)
# # print(loaded_MM)
# energy = calculateEnergy(loaded_MM)
# os = calculate_overshoot(loaded_pitch_angle_sim, -20)
# print(settling_time_pitch, rise_time_depth, energy, os)



# MPC_no_Disturbance_Data = np.load('simulation_data_MPCnoDisturbance0.3.npz')

# loaded_depth_sim = MPC_no_Disturbance_Data['depth_sim']
# loaded_pitch_angle_sim = MPC_no_Disturbance_Data['pitch_angle_sim']
# loaded_BE = MPC_no_Disturbance_Data['BE_input']
# loaded_MM = MPC_no_Disturbance_Data['MM_input']
# # print(loaded_BE)

# # print(loaded_depth_sim)
# # print(loaded_pitch_angle_sim)

# settling_time_depth = calculate_settling_time(loaded_depth_sim, 20, 0.05)
# rise_time_depth = calculate_rise_time(loaded_depth_sim, 20, 0.9)
# energy = calculateEnergy(loaded_BE)
# os = calculate_overshoot(loaded_depth_sim, 20)
# print(settling_time_depth, rise_time_depth, energy, os)

# settling_time_pitch = calculate_settling_time(loaded_pitch_angle_sim, -20, 0.05)
# rise_time_depth = calculate_rise_time(loaded_pitch_angle_sim, -20, 0.9)
# energy = calculateEnergy(loaded_MM)
# os = calculate_overshoot(loaded_pitch_angle_sim,-20)
# # print(loaded_MM)
# print(settling_time_pitch, rise_time_depth, energy,os)


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