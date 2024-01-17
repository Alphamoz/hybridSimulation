from gekko import GEKKO
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pickle
from sklearn.preprocessing import MinMaxScaler

# Load the data
data = pd.read_csv('./data/lastexpguiupdated.csv')
# input data
inputbe = data["inputbe"]
inputmm = data["inputmm"]

# output data
depth_rate_meas = data["field.depth_rate"]
pitch_meas = data["field.pitch_data"]
depth_meas = data["field.depth_data"]

# estimated data
estimated_pitch = data["field.estimated_pitch"]
estimated_pitch=np.append([0,0,0,0], estimated_pitch)
estimated_depth_rate = data["field.estimated_depth_rate"]
estimated_depth_rate=np.append([0,0,0,0], estimated_depth_rate)
estimated_depth_rate /=1000
# setpoint data
depth_rate_sp = data["field.depth_rate_sp"]
pitch_sp = data["field.pitch_sp"]
depth_sp = data["field.depth_sp"]
time1 = data["time1"]
time2 = data["time2"]



# # Print the total error

plt.clf()
plt.suptitle("MPC Control Implementation", fontsize=14, fontweight="bold")
plt.subplot(4, 1, 1)

plt.plot(time1, inputmm,
         label=r'MM', color='red')
# plt.ylabel('Input Signal')
# plt.get
plt.plot(time1, inputbe,
         label='BE', color='blue')
plt.ylabel('Input')
plt.legend(bbox_to_anchor=(1.05, 1.0), ncol=1,
           loc="upper left")

plt.subplot(4, 1, 2)
plt.plot(time1, depth_meas,
         label='Measured', color='royalblue')
plt.plot(time2, depth_sp,
         label=r'Setpoint',linestyle="--", color='orange')
plt.ylabel('Depth (m)')
plt.legend(bbox_to_anchor=(1.05, 1.0), ncol=1,
           loc="upper left")

plt.subplot(4, 1, 3)
plt.plot(time1, pitch_meas,
         label='Measured Pitch', color='royalblue')
plt.plot(time2, pitch_sp,
         label=r'Setpoint',linestyle="--", color='orange')
# plt.plot(time2, estimated_pitch[0:-4],
#          label=r'Estimated',linestyle="--", color='black')
plt.ylabel('Pitch Angle (degree)')
plt.legend(bbox_to_anchor=(1.05, 1.0), ncol=1,
           loc="upper left")

plt.subplot(4, 1, 4)  # Add a new subplot for depth rate
plt.plot(time1, depth_rate_meas,
         label='Measured', color='royalblue')
plt.plot(time2, depth_rate_sp,
         label=r'Setpoint',linestyle="--", color='orange')
# plt.plot(time2, estimated_depth_rate[0:-4],
#          label=r'Estimated',linestyle="--", color='black')
plt.xlabel('Time (s)')
plt.ylabel('Depth Rate (m/s)')
plt.legend(bbox_to_anchor=(1.05, 1.0), ncol=1,
           loc="upper left")

plt.draw()
# plt.pause(0.05)
# plt.tight_layout()
plt.show()
