import numpy as np
import matplotlib.pyplot as plt

MPC_dr_simulation1 = np.load('simulation_data_MPCnoDisturbance1111.npz')
MPC_dr_simulation2 = np.load('simulation_data_MPCnoDisturbance110.10.1.npz')
MPC_dr_simulation3 = np.load('simulation_data_MPCnoDisturbance110.010.01.npz')
MPC_dr_simulation4 = np.load('simulation_data_MPCnoDisturbance110.0010.001.npz')
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
plt.suptitle("MPC Control Simulation Various Weight Input (w)", fontsize=14, fontweight="bold")
plt.subplot(4, 2, 1)

plt.plot(time, MPC_dr_simulation1['MM_input'],
         label=r'MM (w = 1)', color='cornflowerblue')
plt.plot(time, MPC_dr_simulation2['MM_input'],
         label=r'MM (w = $10^{-1}$)', color='indianred')
plt.plot(time, MPC_dr_simulation3['MM_input'],
         label=r'MM (w = $10^{-2}$)', color='lime')
plt.plot(time, MPC_dr_simulation4['MM_input'],
         label=r'MM (w = $10^{-3}$)', color='orange')
plt.ylabel('Input Signal')
# plt.get
plt.legend(bbox_to_anchor=(2.25, 0.5), ncol=2,
           loc="upper left")
plt.subplot(4, 2, 2)
plt.plot(time, MPC_dr_simulation1['BE_input'],
         label='BE (w = 1)', color='navy')
plt.plot(time, MPC_dr_simulation2['BE_input'],
         label=r'BE (w = $10^{-1}$)', color='darkred')
plt.plot(time, MPC_dr_simulation3['BE_input'],
         label=r'BE (w = $10^{-2}$)', color="darkgreen")
plt.plot(time, MPC_dr_simulation4['BE_input'],
         label=r'BE (w = $10^{-3}$)', color='peru')
# plt.ylabel('Input')
plt.legend(bbox_to_anchor=(1.05, 1.0), ncol=2,
           loc="upper left")

plt.subplot(4, 2, (3,4))
plt.plot(time, MPC_dr_simulation1['depth_sim'],
         label='Actual (w = 1)', color='royalblue')
plt.plot(time, MPC_dr_simulation2['depth_sim'],
         label=r'Actual (w = $10^{-1}$)', color='orange')
plt.plot(time, MPC_dr_simulation3['depth_sim'],
         label=r'Actual (w = $10^{-2}$)', color='g')
plt.plot(time, MPC_dr_simulation4['depth_sim'],
         label=r'Actual (w = $10^{-3}$)', color='purple')
plt.axhline(10, color='r', linestyle='--', label='Setpoint Depth')
plt.ylabel('Depth (m)')
plt.legend(bbox_to_anchor=(1.05, 1.0), ncol=1,
           loc="upper left")

plt.subplot(4, 2, (5,6))
plt.plot(time, MPC_dr_simulation1['pitch_angle_sim'],
         label=r'Actual (w = 1)', color='royalblue')
plt.plot(time, MPC_dr_simulation2['pitch_angle_sim'],
         label=r'Actual (w = $10^{-1}$)', color='orange')
plt.plot(time, MPC_dr_simulation3['pitch_angle_sim'],
        label=r'Actual (w = $10^{-2}$)', color='g')
plt.plot(time, MPC_dr_simulation4['pitch_angle_sim'],
         label=r'Actual (w = $10^{-3}$)', color='purple')
plt.axhline(y=-20, color='r',
            linestyle='--', label='Setpoint Pitch')
# plt.xlabel('Time (s)')
plt.ylabel('Pitch Angle (degree)')
plt.legend(bbox_to_anchor=(1.05, 1.0), ncol=1,
           loc="upper left")

plt.subplot(4, 2, (7,8))  # Add a new subplot for depth rate
plt.plot(time, MPC_dr_simulation1['depth_rate_sim'],
         color='royalblue', label=r'Actual (w = 1)')
plt.plot(time, MPC_dr_simulation1['setpointdepth_rate_sim'], color='royalblue',
         linestyle='--', label=r'Setpoint (w = 1)')
plt.plot(time, MPC_dr_simulation2['depth_rate_sim'],
         color='orange', label=r'Actual (w = $10^{-1}$)')
plt.plot(time, MPC_dr_simulation2['setpointdepth_rate_sim'], color='orange',
         linestyle='--', label=r'Setpoint (w = $10^{-1}$)')
plt.plot(time, MPC_dr_simulation3['depth_rate_sim'],
         color='g', label=r'Actual (w = $10^{-2}$)')
plt.plot(time, MPC_dr_simulation3['setpointdepth_rate_sim'], color='g',
         linestyle='--', label=r'Setpoint (w = $10^{-2}$)')
plt.plot(time, MPC_dr_simulation4['depth_rate_sim'],
         color='purple', label=r'Actual (w = $10^{-3}$)')
plt.plot(time, MPC_dr_simulation4['setpointdepth_rate_sim'], color='purple',
         linestyle='--', label=r'Setpoint (w = $10^{-3}$)')
plt.xlabel('Time (s)')
plt.ylabel('Depth Rate \n (mm/s)')
plt.legend(bbox_to_anchor=(1.05, 1.0), ncol=1,
           loc="upper left")

plt.draw()
# plt.pause(0.05)
# plt.tight_layout()
plt.show()
