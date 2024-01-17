import numpy as np
import matplotlib.pyplot as plt

MPC_dr_simulation1 = np.load('simulation_data_MPCnoDisturbance1001510000.npz')
PID_dr_simulation1 = np.load('simulation_data_PIDnoDisturbance10010000.npz')
# MPC_dr_simulation3 = np.load('simulation_data_MPCnoDisturbance3004.npz')
# MPC_dr_simulation4 = np.load('simulation_data_MPCnoDisturbance4004.npz')
# MPC_dr_simulation1


def deleteLastRow(file, lastData = -1):
    modified_data = {}
    for field_name in file.files:
        print (field_name)
        # Get the data matrix for the current field
        data_matrix = file[field_name]

        # Remove the last row from the data matrix
        try:
            print(data_matrix[0:lastData])
            data_matrix = data_matrix[0:lastData]
        except:
            data_matrix = data_matrix
        # if (field_name == "MM_input"):
            # data_matrix = data_matrix*20
        if (field_name == "depth_sim"):
            data_matrix = data_matrix/1000

        # Add the modified data to the dictionary
        modified_data[field_name] = data_matrix
    return modified_data


MPC_dr_simulation1 = deleteLastRow(MPC_dr_simulation1)
PID_dr_simulation1 = deleteLastRow(PID_dr_simulation1,-2)

num_steps = 198
time = np.linspace(0, num_steps, num_steps + 1)

plt.clf()
plt.subplot(4, 1, 1)
plt.title("MPC Control Simulation Various Depth Rate")
plt.plot(time, MPC_dr_simulation1['MM_input'],
         label='MM (MPC)',linestyle='solid', color='magenta')
plt.plot(time, PID_dr_simulation1['MM_input'],
         label='MM (PID)',linestyle='solid', color='green')
plt.plot(time, MPC_dr_simulation1['BE_input'],
         label='BE (MPC)', color='blue')
plt.plot(time, PID_dr_simulation1['BE_input'],
         label='BE (PID)', color='red')

plt.ylabel('Input')
plt.legend(bbox_to_anchor=(1.05, 1.0), ncol=2,
           loc="upper left")

plt.subplot(4, 1, 2)
plt.plot(time, MPC_dr_simulation1['depth_sim'],
         label='Depth (MPC)', color='blue')
plt.plot(time, PID_dr_simulation1['depth_sim'],
         label='Depth (PID)', color='orange')
plt.axhline(10, color='r', linestyle='--', label='Setpoint Depth')
plt.ylabel('Depth')
plt.legend(bbox_to_anchor=(1.05, 1.0), ncol=1,
           loc="upper left")

plt.subplot(4, 1, 3)
plt.plot(time, MPC_dr_simulation1['pitch_angle_sim'],
         label='Pitch Angle (MPC)', color='b')
plt.plot(time, PID_dr_simulation1['pitch_angle_sim'],
         label='Pitch Angle (PID)', color='orange')
plt.axhline(y=-20, color='r',
            linestyle='--', label='Setpoint Pitch')
# plt.xlabel('Time (s)')
plt.ylabel('Pitch Angle')
plt.legend(bbox_to_anchor=(1.05, 1.0), ncol=1,
           loc="upper left")

plt.subplot(4, 1, 4)  # Add a new subplot for depth rate
plt.plot(time, MPC_dr_simulation1['depth_rate_sim'],
         color='b', label='Depth Rate (MPC)')
plt.plot(time, MPC_dr_simulation1['setpointdepth_rate_sim'], color='b',
         linestyle='--', label='Setpoint Depth Rate (MPC)')
plt.plot(time, PID_dr_simulation1['depth_rate_sim'],
         color='red', label='Depth Rate (PID)')
plt.plot(time, PID_dr_simulation1['setpointdepth_rate_sim'], color='red',
         linestyle='--', label='Setpoint Depth Rate (PID)')

# plt.plot(time, MPC_dr_simulation4['depth_rate_sim'],
        #  color='r', label='Depth Rate (0.4 m/s)')
# plt.plot(time, MPC_dr_simulation4['setpointdepth_rate_sim'], color='r',
        #  linestyle='--', label='Setpoint Depth Rate (0.4 m/s)')
plt.xlabel('Time (s)')
plt.ylabel('Depth Rate')
plt.legend(bbox_to_anchor=(1.05, 1.0), ncol=1,
           loc="upper left")

# plt.draw()
# plt.pause(0.05)
plt.tight_layout()
plt.show()
