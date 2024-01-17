import numpy as np
import pandas as pd


# num_steps = 298

# time = np.linspace(0, num_steps, num_steps + 1)

def calculate_settling_time(data, setpoint, percentage, time):
    # print("ini datanya",np.array(data)-setpoint)
    settling_start_index = np.argmax(np.abs(data - setpoint) <= abs(setpoint*percentage))
    # print("settlingtime bos!",time[settling_start_index])
    if settling_start_index > 0:
        settling_time = time[settling_start_index]
    else:
        settling_time = np.nan
    return settling_time

def calculate_rise_time(data, setpoint, threshold, time):
    rise_start_index = np.argmax((data * setpoint/abs(setpoint)) >= abs(threshold*setpoint*setpoint/abs(setpoint)))
    # print("Rise Time BOS!", np.abs(data-setpoint) <= abs(threshold*setpoint))
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
    # print("settling point",settlingpoint)
    dataAfterSettling = data[int(settlingpoint):]
    meanError = np.mean(np.abs(dataAfterSettling-setpoint))
    return meanError

for first1,first2,last1,last2 in [[0,0,-1,-1]]:
    data = pd.read_csv('./data/dataProcessing/processedData/MPCPIDExperimental.csv')
    # setpoint data
    print("MPC dulu")
    # depth_rate_sp = data["field.depth_rate_sp"][first1:last1]
    pitch_sp = data["pitch_sp"][first1:last1]
    depth_sp = data["depth_sp"][first1:last1]
    time1 = data["time1"]
    # time2 = data["time2"]
    # print(np.argmin(np.array(time1)<500))
    # print(np.argmin(np.array(time2)<500))
    
    loaded_depth_sim = data["depth_mpc"][first2:last2]
    loaded_pitch_angle_sim = data["pitch_mpc"][first2:last2]
    # print(loaded_pitch_angle_sim)
    
    loaded_BE = data["inputbe_mpc"][first2:last2]
    loaded_MM = data["inputmm_mpc"][first2:last2]
    # loaded_depth_rate_sim =data["field.depth_rate"][first2:last2]
    # setpointdepth_rate_sim = data["field.depth_rate_sp"][first2:last2]

    settling_time_depth = calculate_settling_time(np.array(loaded_depth_sim), 3.5, 0.05,time1)
    rise_time_depth = calculate_rise_time(loaded_depth_sim, 3.5, 0.9,time1)
    energy = calculateEnergy(loaded_BE)
    max_os_depth, os_depth = calculate_overshoot(loaded_depth_sim,3.5)
    steady_error = calculate_steady_error(loaded_depth_sim, 3.5, rise_time_depth)
    std_deviation_BE = np.std(loaded_BE)
    std_deviation_MM = np.std(loaded_MM)
    print("Depth for {}".format(first1), settling_time_depth, rise_time_depth, energy, os_depth, steady_error, "with max os", max_os_depth)

    settling_time_pitch = calculate_settling_time(loaded_pitch_angle_sim, -10, 0.05, time1)
    rise_time_pitch = calculate_rise_time(loaded_pitch_angle_sim, -10, 0.9, time1)
    energy = calculateEnergy(loaded_MM)
    max_os_pitch, os_pitch = calculate_overshoot(loaded_pitch_angle_sim, -10)
    steady_error = calculate_steady_error(loaded_pitch_angle_sim, -10, settling_time_pitch)
    print("Pitch for {}".format(first1),settling_time_pitch, rise_time_pitch, energy, os_pitch, steady_error, "with max os", max_os_pitch)
    
    
    print("Baru PID")
    # depth_rate_sp = data["field.depth_rate_sp"][first1:last1]
    pitch_sp = data["pitch_sp"][first1:last1]
    depth_sp = data["depth_sp"][first1:last1]
    time1 = data["time1"]
    # time2 = data["time2"]
    # print(np.argmin(np.array(time1)<500))
    # print(np.argmin(np.array(time2)<500))
    
    loaded_depth_sim = data["depth_pid"][first2:last2]
    loaded_pitch_angle_sim = data["pitch_pid"][first2:last2]
    # print(loaded_pitch_angle_sim)
    
    loaded_BE = data["inputbe_pid"][first2:last2]
    loaded_MM = data["inputmm_pid"][first2:last2]
    # loaded_depth_rate_sim =data["field.depth_rate"][first2:last2]
    # setpointdepth_rate_sim = data["field.depth_rate_sp"][first2:last2]

    settling_time_depth = calculate_settling_time(np.array(loaded_depth_sim), 3.5, 0.05,time1)
    rise_time_depth = calculate_rise_time(loaded_depth_sim, 3.5, 0.9,time1)
    energy = calculateEnergy(loaded_BE)
    max_os_depth, os_depth = calculate_overshoot(loaded_depth_sim,3.5)
    steady_error = calculate_steady_error(loaded_depth_sim, 3.5, rise_time_depth)
    std_deviation_BE = np.std(loaded_BE)
    std_deviation_MM = np.std(loaded_MM)
    print("Depth for {}".format(first1), settling_time_depth, rise_time_depth, energy, os_depth, steady_error, "with max os", max_os_depth)

    settling_time_pitch = calculate_settling_time(loaded_pitch_angle_sim, -10, 0.05, time1)
    rise_time_pitch = calculate_rise_time(loaded_pitch_angle_sim, -10, 0.9, time1)
    energy = calculateEnergy(loaded_MM)
    max_os_pitch, os_pitch = calculate_overshoot(loaded_pitch_angle_sim, -10)
    steady_error = calculate_steady_error(loaded_pitch_angle_sim, -10, settling_time_pitch)
    print("Pitch for {}".format(first1),settling_time_pitch, rise_time_pitch, energy, os_pitch, steady_error, "with max os", max_os_pitch)
    
    # settling_time_depth_rate = 0
    # rise_time_depth_rate = 0
    # # settling_time_depth_rate = calculate_settling_time(loaded_depth_rate_sim, depth_rate_sp, 0.05,time1)
    # # rise_time_depth_rate = calculate_rise_time(loaded_depth_rate_sim, depth_rate_sp, 0.9,time1)
    # max_os_dr,os_dr = calculate_overshoot(loaded_depth_rate_sim, 0.1)
    # steady_error = calculate_steady_error(loaded_depth_rate_sim, setpointdepth_rate_sim[int(50):], 50)

    # print("Depth Rate for {}".format(first1),settling_time_depth_rate, rise_time_depth_rate, energy, os_dr,steady_error, "with max os", max_os_dr)
    # print("Tulis gan!",rise_time_depth, rise_time_pitch, rise_time_depth_rate,os_depth,os_pitch, os_dr ,std_deviation_BE,std_deviation_MM)