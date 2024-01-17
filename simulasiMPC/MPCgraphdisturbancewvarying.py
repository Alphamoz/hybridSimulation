import numpy as np
import matplotlib.pyplot as plt

MPC_dr_simulation1 = np.load('simulation_data_MPCnoDisturbanceweightvaryingFalse.npz')
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


num_steps = 298
time = np.linspace(0, num_steps, num_steps + 1)

plt.clf()
plt.subplot(4, 1, 1)
plt.title("MPC Control Simulation Various Depth Rate")
plt.plot(time, MPC_dr_simulation1['MM_input'],
         label='MM')
plt.plot(time, MPC_dr_simulation1['BE_input'],
         label='BE')
plt.ylabel('Input')
plt.legend(bbox_to_anchor=(1.05, 1.0), ncol=1,
           loc="upper left")

plt.subplot(4, 1, 2)
plt.plot(time, MPC_dr_simulation1['depth_sim'],
         label='Actual')
plt.axhline(20, color='r', linestyle='--', label='Setpoint')
plt.ylabel('Depth (m)')
plt.legend(bbox_to_anchor=(1.05, 1.0), ncol=1,
           loc="upper left")

plt.subplot(4, 1, 3)
plt.plot(time, MPC_dr_simulation1['pitch_angle_sim'],
         label='Actual')
plt.axhline(y=-20, color='r',
            linestyle='--', label='Setpoint')
# plt.xlabel('Time (s)')
plt.ylabel('Pitch Angle \n (degree)')
plt.legend(bbox_to_anchor=(1.05, 1.0), ncol=1,
           loc="upper left")

plt.subplot(4, 1, 4)  # Add a new subplot for depth rate
plt.plot(time, MPC_dr_simulation1['depth_rate_sim'],
        label='Actual')
plt.plot(time, MPC_dr_simulation1['setpointdepth_rate_sim'], color='r',
         linestyle='--', label='Setpoint')
plt.xlabel('Time (s)')
plt.ylabel('Depth Rate \n (mm/s)')
plt.legend(bbox_to_anchor=(1.05, 1.0), ncol=1,
           loc="upper left")

# plt.draw()
# plt.pause(0.05)
plt.tight_layout()
plt.show()
