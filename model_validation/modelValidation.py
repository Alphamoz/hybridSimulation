from gekko import GEKKO
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pickle

with open('p_mm_pitch.pickle', 'rb') as file:
    p_mm_pitch = pickle.load(file)
    
with open('p_be_dr.pickle', 'rb') as file:
    p_be_dr = pickle.load(file)
    
with open('p_be_pitch.pickle', 'rb') as file:
    p_be_pitch = pickle.load(file)

time_to_simulate = 10  # Time to simulate (seconds)
sampling_time = 1
time_steps = int(time_to_simulate / (sampling_time))  # 

# data validation
data = pd.read_csv('./data/dataProcessing/processedData/downsampled_data2.csv')
# data = pd.read_csv('./data/dataProcessing/processedData/DataModel5detik.csv')
# input1 = "be_rate"
input1 = "field.internal_volume"
input2 = "field.cur_mm"
# input3 = "field.internal_volume"
output1 = "field.depth_rate"
output2 = "field.pitch_data"
# input1 = "BE"
# input2 = "MM"
# output1 = "Depth_Rate"
# output2 = "Pitch"
t = data['Time']
u = data.loc[:,[input1, input2]]
y = data.loc[:,[output1, output2]]
# offset = 9
startData = 10
u = u[startData:150]
y = y[startData:150]
# print(u.shape)
t = t[0:u.shape[0]]
# u = u[10:100]
# y = y[10:100]
# print(u.shape)
u[input1][:] -= 300
u[input2][:] -= 250
# y[output1][:] *= 1
y[output1][:] *= 1000

m = GEKKO(remote=False)
#       dynamics dr, dynamics pitch
# a->   [a1(k-1), a2(k-1)]
#       [a1(k-2),a2(k-2)]   etc...

# dynamics input BE, dynamics input MM for dr
# dynamics input BE, dynamics input MM for pitch
# b -> b for BE[[b1(k-1), b2(k-1)], 
#               [b1(k-2), b2(k-2)]
#   -> b for MM[[b1(k-1), b2(k-1)], [b1(k-2), b2(k-2)] 
p = {'a': np.array([[ p_be_dr['a'][0][0], p_be_pitch['a'][0][0]],
       [ p_be_dr['a'][1][0] , p_be_pitch['a'][1][0]],
       [-0.00,0]]), 'b': np.array([[[ p_be_dr['b'][0][0][0],  9.15255674e-09],
                                         [0, 0]],

       [[p_be_pitch['b'][0][0][0],  p_mm_pitch['b'][0][0][0]],
        [0, 0]
        ]]), 'c': np.array([ p_be_dr['c'][0], p_be_pitch['c'][0]])}

print(p)
yc, uc = m.arx(p)
m.options.IMODE = 1
# yc[1].value = 9
m.solve(disp=False)

for start_point in range(10, len(u), 10):  # Start at data point 10, increment by 10 until the end of the data
    start_time = t[start_point]  # Get the time corresponding to the start point
    end_time = start_time + time_to_simulate  # Calculate the end time for simulation
    
    # Extract a segment of data for simulation
    u_sim = u[0:start_point + time_steps].reset_index(drop=True)
    y_sim = y[0:start_point + time_steps].reset_index(drop=True)
    # y_sim.reset_index(drop=True)
    print(y_sim)
    m.time = np.linspace(0, start_point+u_sim.shape[0]-1, +start_point+time_steps)
    print(m.time)
    # Set initial conditions for the simulation
    # m.time = np.linspace(0, time_steps-1, time_steps)
    uc[0].value = u_sim[input1]
    uc[1].value = u_sim[input2]
    yc[0].value = y_sim[output1][0]
    yc[1].value = y_sim[output2][0] 
    
    # Solve the model for simulation
    m.options.TIME_SHIFT=0
    m.options.IMODE = 4
    m.solve(disp=False)
    
    # Get the simulated output
    
    yc_sim = np.array(yc).T
    mse = np.mean((yc - y)**2, axis=0)
    print("Total error (MSE):", mse)
    titlesize = 12  
    fontsize = 12
    print(yc_sim.T[0])
    # Plot the results
    plt.clf()
    plt.subplot(2, 1, 1)
    plt.title('ARX Model Validation', fontsize=titlesize , fontweight='bold')
    plt.plot(m.time, uc[0], 'b-', label='BE')
    plt.plot(m.time, uc[1], 'r-', label='MM')
    plt.ylabel('Actuator Value', fontsize=fontsize)
    plt.xticks(fontsize=fontsize)
    plt.yticks(fontsize=fontsize)
    plt.legend(bbox_to_anchor=(1.05, 1.0), ncol=1,
           loc="upper left", fontsize=fontsize)


    plt.subplot(2, 1, 2)
    plt.plot(m.time, y_sim[output1], 'b-',
            label='Depth Rate Measured (mm/s)')
    plt.plot(m.time, yc_sim.T[0], 'g:',
            label=r'Depth Rate Predicted Test')
    # plt.plot(m.time, yc[0], 'g:',
    #          label=r'Depth Rate Predicted Test', linewidth=2)

    plt.plot(m.time, y_sim[output2], 'c-', label='Pitch Measured (degree)')
    plt.plot(m.time, yc_sim.T[1], 'r:', label=r'Pitch Predicted Test')
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
    # plt.show()



# # simulation
# m.options.TIME_SHIFT=0
# m.options.IMODE = 4
# uc[0].value = u[input1]
# uc[1].value = u[input2]
# # uc[2].value = u[input3]
# # print(y[output1][0])
# # print(y[output2][0])
# yc[0].value = y[output1][startData]
# yc[1].value = y[output2][startData] 
# # mm = u[input2] * 20

# m.solve(disp=False)

# print(type(y))
# # print(type(yc))

# yc = np.array(yc).T
# # yc.T[1]+=offset
# # print(yc_array-y)

plt.show()

