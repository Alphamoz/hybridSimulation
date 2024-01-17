from gekko import GEKKO
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pickle
from sklearn.preprocessing import MinMaxScaler

# Load the data
data = pd.read_csv('./data/dataProcessing/processedData/MPCPIDExperimental.csv')
# input data
inputbe_mpc = data["inputbe_mpc"]
inputmm_mpc = data["inputmm_mpc"]
inputbe_pid = data["inputbe_pid"]
inputmm_pid = data["inputmm_pid"]

sumbe_mpc = data["inputbe_sum_mpc"]
summm_mpc = data["inputmm_sum_mpc"]
sumbe_pid = data["inputbe_sum_pid"]
summm_pid = data["inputmm_sum_pid"]

# output data
# depth_rate_meas = data["field.depth_rate"]
pitch_mpc = data["pitch_mpc"]
depth_mpc = data["depth_mpc"]
pitch_pid = data["pitch_pid"]
depth_pid = data["depth_pid"]

# # estimated data
# estimated_pitch = data["field.estimated_pitch"]
# estimated_pitch=np.append([0,0,0,0], estimated_pitch)
# estimated_depth_rate = data["field.estimated_depth_rate"]
# estimated_depth_rate=np.append([0,0,0,0], estimated_depth_rate)
# estimated_depth_rate /=1000
# setpoint data
# depth_rate_sp = data["field.depth_rate_sp"]
pitch_sp = data["pitch_sp"]
depth_sp = data["depth_sp"]
time1 = data["time1"]
# time2 = data["time2"]



# # Print the total error

plt.clf()
plt.suptitle("MPC vs PID Experimental", fontsize=14, fontweight="bold")
plt.subplot(4, 1, 1)

plt.plot(time1, inputmm_pid,
         label=r'MM PID', color='orangered')
plt.plot(time1, inputmm_mpc,
         label=r'MM MPC', color='royalblue')
# plt.ylabel('Input Signal')
# plt.get
plt.plot(time1, inputbe_pid,
         label='BE PID', color='limegreen')
plt.plot(time1, inputbe_mpc,
         label='BE MPC', color='darkviolet')
plt.ylabel('Input')
plt.legend(bbox_to_anchor=(1.05, 1.0), ncol=1,
           loc="upper left")

plt.subplot(4, 1, 2)
plt.plot(time1, depth_pid,
         label='PID', color='limegreen')
plt.plot(time1, depth_mpc,
         label='MPC', color='royalblue')
plt.plot(time1, depth_sp,
         label=r'Setpoint',linestyle="--", color='red')
plt.ylabel('Depth\n(m)')
plt.legend(bbox_to_anchor=(1.05, 1.0), ncol=1,
           loc="upper left")

plt.subplot(4, 1, 3)
plt.plot(time1, pitch_pid,
         label='PID', color='limegreen')
plt.plot(time1, pitch_mpc,
         label='MPC', color='royalblue')
plt.plot(time1, pitch_sp,
         label=r'Setpoint',linestyle="--", color='red')
# plt.plot(time2, estimated_pitch[0:-4],
#          label=r'Estimated',linestyle="--", color='black')
plt.ylabel('Pitch Angle\n(degree)')
plt.legend(bbox_to_anchor=(1.05, 1.0), ncol=1,
           loc="upper left")

plt.subplot(4, 1, 4)  # Add a new subplot for depth rate
plt.plot(time1, sumbe_pid,
         label='BE PID', color='limegreen')
plt.plot(time1, sumbe_mpc,
         label=r'BE MPC', color='darkviolet')
plt.plot(time1,summm_pid,label='MM PID', color="orangered")
plt.plot(time1,summm_mpc,label='MM MPC', color="royalblue")
# plt.plot(time2, estimated_depth_rate[0:-4],
#          label=r'Estimated',linestyle="--", color='black')
plt.xlabel('Time (s)')
plt.ylabel('Total Input\nChanges')
plt.legend(bbox_to_anchor=(1.05, 1.0), ncol=1,
           loc="upper left")

plt.draw()
# plt.pause(0.05)
# plt.tight_layout()
plt.show()
