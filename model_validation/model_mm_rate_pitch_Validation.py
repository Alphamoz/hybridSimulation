from gekko import GEKKO
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pickle
    
with open('p_mm_pitch.pickle', 'rb') as file:
    p = pickle.load(file)

time_to_simulate = 20  # Time to simulate (seconds)
sampling_time = 1
time_steps = int(time_to_simulate / (sampling_time))  # 

# data validation
data = pd.read_csv('./data/dataProcessing/processedData/DataModel5detik.csv')
input1 = "mm_rate"
output1 = "pitch"
t = data['Time']
u = data.loc[:,[input1]]
y = data.loc[:,[output1]]

startData = 0

u = u[startData:].reset_index(drop=True)
y = y[startData:].reset_index(drop=True)
t = t[0:u.shape[0]]

m = GEKKO(remote=False)
#       dynamics dr, dynamics pitch
# a->   [a1(k-1), a2(k-1)]
#       [a1(k-2),a2(k-2)]   etc...

# dynamics input BE, dynamics input MM for dr
# dynamics input BE, dynamics input MM for pitch
# b -> b for BE[[b1(k-1), b2(k-1)], 
#               [b1(k-2), b2(k-2)]
#   -> b for MM[[b1(k-1), b2(k-1)], [b1(k-2), b2(k-2)] 
# p = {'a': np.array([[ p_be_dr['a'][0][0]],
#        [ 0 ],
#        [-0.00]]), 'b': np.array([[[ p_be_dr['b'][0][0][0]],
#                                 [0],
#                                          ],])
#        , 'c': np.array([ p_be_dr['c'][0]])}

print(p)
yc, uc = m.arx(p)
m.options.IMODE = 1
m.solve(disp=False)
ypred_forplot = []
u_sim_forplot = []
t_sim_forplot = []

for start_point in range(0, len(u), time_to_simulate):  # Start at data point 10, increment by 10 until the end of the data
    start_time = t[start_point]  # Get the time corresponding to the start point
    end_time = start_time + time_to_simulate  # Calculate the end time for simulation
    
    # Extract a segment of data for simulation
    u_sim = u[start_point:start_point + time_steps].reset_index(drop=True)
    y_sim = y[start_point:start_point + time_steps].reset_index(drop=True)
    # y_sim.reset_index(drop=True)
    print(y_sim)
    # Set initial conditions for the simulation
    m.time = np.linspace(0, time_steps-1, time_steps)
    uc[0].value = u_sim[input1]
    yc[0].value = y_sim[output1][0]
    
    # Solve the model for simulation
    # m.options.TIME_SHIFT=0
    m.options.IMODE = 4
    m.solve(disp=False)
    
    # Get the simulated output 
    ypred = np.array(yc).T
    # adding the values only of array to the list using extend
    ypred_forplot.extend(ypred)
    u_sim_forplot.extend(u_sim[input1])
    t_sim_forplot.extend(np.linspace(start_point, start_point+time_steps-1,time_steps))
    print(t_sim_forplot,u_sim_forplot)
    # getting MSE
    mse = np.mean((yc - y)**2, axis=0)
    print("Total error (MSE):", mse)
    
    titlesize = 12  
    fontsize = 12
    # Plot the results
    plt.clf()
    plt.subplot(2, 1, 1)
    plt.title('ARX Model Validation', fontsize=titlesize , fontweight='bold')
    plt.plot(t_sim_forplot, u_sim_forplot, 'b-', label='BE_rate')
    # plt.plot(m.time, uc[1], 'r-', label='MM')
    plt.ylabel('Actuator Value', fontsize=fontsize)
    plt.xticks(fontsize=fontsize)
    plt.yticks(fontsize=fontsize)
    plt.legend(bbox_to_anchor=(1.05, 1.0), ncol=1,
           loc="upper left", fontsize=fontsize)


    plt.subplot(2, 1, 2)
    plt.plot(t_sim_forplot, y[output1][0:start_point+time_steps], 'b-',
            label='Depth Rate Measured (mm/s)')
    plt.plot(t_sim_forplot, ypred_forplot, 'g:',
            label=r'Depth Rate Predicted Test')
    # plt.plot(m.time, yc[0], 'g:',
    #          label=r'Depth Rate Predicted Test', linewidth=2)

    # plt.plot(m.time, y_sim[output2], 'c-', label='Pitch Measured (degree)')
    # plt.plot(m.time, yc_sim.T[1], 'r:', label=r'Pitch Predicted Test')
    # plt.plot(m.time, yc[1], 'r:',
    #          label=r'$Pitch Predicted Test$', linewidth=2)
    plt.ylabel('Output', fontsize=fontsize)
    plt.xlabel('Time(s)', fontsize=fontsize)
    plt.legend(bbox_to_anchor=(1.05, 1.0), ncol=1,
            loc="upper left", fontsize=fontsize)
    plt.xticks(fontsize=fontsize)
    plt.yticks(fontsize=fontsize)
    plt.tight_layout()
    plt.savefig("ValidationData.png")
    plt.draw()
    plt.pause(0.05)
    plt.show()



# # simulation
# m.options.TIME_SHIFT=0
# m.options.IMODE = 4
# m.time= np.linspace(0, u.shape[0]-1, u.shape[0])
# uc[0].value = u[input1]
# yc[0].value[0] = y[output1][0]
# print(yc[0].value[0])

# m.solve(disp=False)

# print(type(y))
# # print(type(yc))

# yc = np.array(yc).T
# # yc.T[1]+=offset
# # print(yc_array-y)

# titlesize = 12  
# fontsize = 12
# # print(yc_sim.T[0])
# # Plot the results
# plt.clf()
# plt.subplot(2, 1, 1)
# plt.title('ARX Model Validation', fontsize=titlesize , fontweight='bold')
# plt.plot(m.time, uc[0], 'b-', label='BE_rate')
# # plt.plot(m.time, uc[1], 'r-', label='MM')
# plt.ylabel('Actuator Value', fontsize=fontsize)
# plt.xticks(fontsize=fontsize)
# plt.yticks(fontsize=fontsize)
# plt.legend(bbox_to_anchor=(1.05, 1.0), ncol=1,
#         loc="upper left", fontsize=fontsize)


# plt.subplot(2, 1, 2)
# print("ini bos", y[output1][startData])
# print("ini bos yang kedua", yc.T[0])
# plt.plot(m.time, y[output1][startData:], 'b-',
#         label='Depth Rate Measured (mm/s)')
# plt.plot(m.time, yc.T[0], 'g:',
#         label=r'Depth Rate Predicted Test')
# # plt.plot(m.time, yc[0], 'g:',
# #          label=r'Depth Rate Predicted Test', linewidth=2)

# # plt.plot(m.time, y_sim[output2], 'c-', label='Pitch Measured (degree)')
# # plt.plot(m.time, yc_sim.T[1], 'r:', label=r'Pitch Predicted Test')
# # plt.plot(m.time, yc[1], 'r:',
# #          label=r'$Pitch Predicted Test$', linewidth=2)
# plt.ylabel('Output', fontsize=fontsize)
# plt.xlabel('Time(s)', fontsize=fontsize)
# plt.legend(bbox_to_anchor=(1.05, 1.0), ncol=1,
#         loc="upper left", fontsize=fontsize)
# plt.xticks(fontsize=fontsize)
# plt.yticks(fontsize=fontsize)
# plt.tight_layout()
# plt.savefig("ValidationData.png")
# plt.draw()
# plt.pause(0.05)
plt.show()

