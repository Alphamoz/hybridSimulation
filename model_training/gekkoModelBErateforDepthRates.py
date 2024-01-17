from gekko import GEKKO
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pickle
from sklearn.preprocessing import MinMaxScaler

# Load the data
# data = pd.read_csv('./data/dataProcessing/processedData/.csv')
# print(data["Time"])
# input1 = "field.internal_volume"
# input2 = "field.cur_mm"
# output1 = "field.depth_rate"
# output2 = "field.pitch_data"
data = pd.read_csv('./data/dataProcessing/processedData/downsampled_data3.csv')
input1 = "be_rate"
output1 = "depth_rate"
t = data['Time']
u = data.loc[:,[input1]]
y = data.loc[:,[output1]]

u = u[20:]
y = y[20:]
# print(u.shape)
t = t[0:u.shape[0]]
# print(u.size, y.size, t.size)

# # print(u_scaled)
# # print(y_scaled)
# u[input2][:] -= 0
# u[input2][:] -= 0
# scaled by 100 (decimeter)
# y[output1][:] -= y[output1][0]
# y[output2][:] *= 1

# print(y[output1].max())

# print(u.shape)
# print(y.shape)
# print(t.shape)
m = GEKKO(remote=False)

na = 1  # output coefficients
nb = 2  # input coefficients
# # depth rate become mm to makes the data visible
# # y.iloc[:, 0] *= 1
# print(u)
# print(y)

yp, p, K = m.sysid(t, u=u,
                   y=y, na=na, nb=nb, pred='meas', shift='none', scale=False)

# # Calculate mean squared error to determine best na and nb
mse = np.mean((yp - y)**2, axis=0)

with open('p_br_dr.pickle', 'wb') as file:
    pickle.dump(p, file)
with open('k_values.pickle', 'wb') as file:
    pickle.dump(K, file)
# # Print the total error
print("Total error (MSE):", mse)
print("p", p)
plt.figure()
plt.subplot(2, 1, 1)
plt.plot(t, u)
plt.title("ARX Model Predicted Output vs Measured Output")
plt.legend(["BE_rate", "MM"])
plt.ylabel('Input Value')
plt.subplot(2, 1, 2)
plt.plot(t, y)
plt.plot(t, yp)
plt.legend(["Depth_rate_meas",
            "Depth_rate_pred", "Pitch_pred"])
plt.ylabel('Output Value')
plt.xlabel('Time')
plt.show()
