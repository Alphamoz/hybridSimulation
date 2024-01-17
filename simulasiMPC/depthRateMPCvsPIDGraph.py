import numpy as np
import matplotlib.pyplot as plt

MPC_dr_simulation1 = np.load('simulation_data_MPCnoDisturbance1004.npz')
MPC_dr_simulation2 = np.load('simulation_data_MPCnoDisturbance2004.npz')
MPC_dr_simulation3 = np.load('simulation_data_MPCnoDisturbance3004.npz')
MPC_dr_simulation4 = np.load('simulation_data_MPCnoDisturbance4004.npz')
# MPC_dr_simulation1


def deleteLastRow(file):
    modified_data = {}
    for field_name in file.files:
        print (field_name)
        # Get the data matrix for the current field
        data_matrix = file[field_name]

        # Remove the last row from the data matrix
        try:
            print(data_matrix[0:-1])
            data_matrix = data_matrix[0:-1]
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
MPC_dr_simulation2 = deleteLastRow(MPC_dr_simulation2)
MPC_dr_simulation3 = deleteLastRow(MPC_dr_simulation3)
MPC_dr_simulation4 = deleteLastRow(MPC_dr_simulation4)


num_steps = 298
time = np.linspace(0, num_steps, num_steps + 1)

plt.clf()
plt.subplot(4, 1, 1)
plt.title("MPC Control Simulation Various Depth Rate")
plt.plot(time, MPC_dr_simulation1['MM_input'],
         label='MM (0.1 m/s)',linestyle='-.', color='cornflowerblue')
plt.plot(time, MPC_dr_simulation2['MM_input'],
         label='MM (0.2 m/s)',linestyle='-.', color='indianred')
plt.plot(time, MPC_dr_simulation3['MM_input'],
         label='MM (0.3 m/s)',linestyle='-.', color='springgreen')
# plt.plot(time, MPC_dr_simulation4['MM_input'],
        #  label='MM (0.4 m/s)', color='indianred')
plt.plot(time, MPC_dr_simulation1['BE_input'],
         label='BE (0.1 m/s)', color='blue')
plt.plot(time, MPC_dr_simulation2['BE_input'],
         label='BE (0.2 m/s)', color='red')
plt.plot(time, MPC_dr_simulation3['BE_input'],
         label='BE (0.3 m/s)', color="green")
# plt.plot(time, MPC_dr_simulation4['BE_input'],
        #  label='BE (0.4 m/s)', color='red')
plt.ylabel('Input')
plt.legend(bbox_to_anchor=(1.05, 1.0), ncol=2,
           loc="upper left")

plt.subplot(4, 1, 2)
plt.plot(time, MPC_dr_simulation1['depth_sim'],
         label='Depth (0.1 m/s)', color='blue')
plt.plot(time, MPC_dr_simulation2['depth_sim'],
         label='Depth (0.2 m/s)', color='orange')
plt.plot(time, MPC_dr_simulation3['depth_sim'],
         label='Depth (0.3 m/s)', color='g')
# plt.plot(time, MPC_dr_simulation4['depth_sim'],
        #  label='Depth (0.4 m/s)', color='red')
plt.axhline(20, color='r', linestyle='--', label='Setpoint Depth')
plt.ylabel('Depth')
plt.legend(bbox_to_anchor=(1.05, 1.0), ncol=1,
           loc="upper left")

plt.subplot(4, 1, 3)
plt.plot(time, MPC_dr_simulation1['pitch_angle_sim'],
         label='Pitch Angle (0.1 m/s)', color='b')
plt.plot(time, MPC_dr_simulation2['pitch_angle_sim'],
         label='Pitch Angle (0.2 m/s)', color='orange')
plt.plot(time, MPC_dr_simulation3['pitch_angle_sim'],
         label='Pitch Angle (0.3 m/s)', color='g')
# plt.plot(time, MPC_dr_simulation4['pitch_angle_sim'],
        #  label='Pitch Angle (0.4 m/s)', color='r')
plt.axhline(y=-20, color='r',
            linestyle='--', label='Setpoint Pitch')
# plt.xlabel('Time (s)')
plt.ylabel('Pitch Angle')
plt.legend(bbox_to_anchor=(1.05, 1.0), ncol=1,
           loc="upper left")

plt.subplot(4, 1, 4)  # Add a new subplot for depth rate
plt.plot(time, MPC_dr_simulation1['depth_rate_sim'],
         color='b', label='Depth Rate (0.1 m/s)')
plt.plot(time, MPC_dr_simulation1['setpointdepth_rate_sim'], color='b',
         linestyle='--', label='Setpoint Depth Rate (0.1 m/s)')
plt.plot(time, MPC_dr_simulation2['depth_rate_sim'],
         color='red', label='Depth Rate (0.2 m/s)')
plt.plot(time, MPC_dr_simulation2['setpointdepth_rate_sim'], color='red',
         linestyle='--', label='Setpoint Depth Rate (0.2 m/s)')
plt.plot(time, MPC_dr_simulation3['depth_rate_sim'],
         color='g', label='Depth Rate (0.3 m/s)')
plt.plot(time, MPC_dr_simulation3['setpointdepth_rate_sim'], color='g',
         linestyle='--', label='Setpoint Depth Rate (0.3 m/s)')
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
