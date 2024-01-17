from gekko import GEKKO
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pickle
import math
import time as myTime
first = True
na_s = [1,2,3,4,5,6]  # output coefficients
nb_s = [1,2,3,4,5,6]  # input coefficients

# function to load the data
def initializeData(data, inputs, outputs, time, startData, endData, normalization = False, scaleOutput = False, zeroing = True):
    data = pd.read_csv(data)
    t = data[time]
    u = data.loc[:,inputs]
    y = data.loc[:,outputs]
    u = u[startData:].reset_index(drop=True) if endData == 0 else u[startData:endData].reset_index(drop=True)
    y = y[startData:].reset_index(drop=True) if endData == 0 else y[startData:endData].reset_index(drop=True)
    t = t[0:u.shape[0]]
    if normalization:
        u[inputs[0]][:] -= normalization
    if zeroing:
        y[outputs[0]][:] -= y[outputs[0]][0]
    if scaleOutput:
        y[outputs[0]][:] *= scaleOutput
    return u, y, t

for na in na_s:
    for nb in nb_s:        
        # be, depth rate model
        u,y,t = initializeData(data='./data/dataProcessing/processedData/DataModel5detik.csv',inputs = ["be_rate"], outputs = ["depth_rate"], time=["Time"],startData=0, endData=200, normalization=0, zeroing=False, scaleOutput=0)
        m = GEKKO(remote=False)
        yp_be_dr, p_be_dr, K_be_dr = m.sysid(t, u=u,
                   y=y, na=na, nb=nb, pred='meas', shift='none', scale=False)
        mse_be_dr = np.mean((yp_be_dr - y)**2, axis=0)
        print(p_be_dr)
        # myTime.sleep(5)

        # # Print the total error
        print("Total error (MSE):", mse_be_dr["depth_rate"])
        mse_dict = {'na': [na],
                    'nb': [nb],
                    'MSE_depthrate':[mse_be_dr["depth_rate"]],
                    }
        print(mse_dict)
        mse_data = pd.DataFrame(mse_dict)
        mse_data.to_csv('mse_values_train_depth_rate.csv', mode='w' if first==True else 'a', header=(True if first==True else False), index=False)

        # data validation
        data = pd.read_csv('../data/dataProcessing/processedData/downsampled_data2.csv')
        input1 = "be_rate"
        output1 = "depth_rate"
        t = data['Time']
        u = data.loc[:,[input1]]
        y = data.loc[:,[output1]]
        startData = 0
        endData = -10
        u = u[startData:endData].reset_index(drop=True)
        y = y[startData:endData].reset_index(drop=True)
        t = t[0:u.shape[0]]
        
        y[output1][:] -= y[output1][0]

        p=p_be_dr
        yc, uc = m.arx(p)
        m.options.IMODE = 1
        m.solve(disp=False)
        mse_result=[]
        
        # doing the simulation perstep
        time_to_simulate = 20  # Time to simulate (seconds)
        sampling_time = 1
        time_steps = int(time_to_simulate / (sampling_time))  # 
        u_sim = u[0:].reset_index(drop=True)
        y_sim = y[0:].reset_index(drop=True)
        # y_sim.reset_index(drop=True)
        print(y_sim)
        # Set initial conditions for the simulation
        m.time = t
        uc[0].value = u_sim[input1]
        # uc[1].value = u_sim[input2]
        yc[0].value = y_sim[output1][0]
        print(yc)
        m.options.IMODE = 4
        m.solve(disp=False)
        # Get the simulated output 
        ypred = np.array(yc).T
        print(ypred[0:,0])
        print(y_sim["depth_rate"])
        print(ypred[0:,0] - y_sim["depth_rate"])
        mse = np.mean((ypred[0:,0] - y_sim["depth_rate"])**2, axis=0)
        print(mse)
        mse_dict = {'na': [na],
                    'nb': [nb],
                    'MSE_pitch':[mse]
                    }
        print(mse_dict)
        mse_data = pd.DataFrame(mse_dict)
        mse_data.to_csv('mse_values_test_depth_rate.csv', mode='w' if first==True else 'a', header=(True if first==True else False), index=False)
        first = False 
        
        # for start_point in range(0, math.floor(len(u)/time_to_simulate)*time_to_simulate, time_to_simulate):  # Start at data point 10, increment by 10 until the end of the data
        #     start_time = t[start_point]  # Get the time corresponding to the start point
        #     end_time = start_time + time_to_simulate  # Calculate the end time for simulation
            
        #     # Extract a segment of data for simulation
        #     u_sim = u[start_point:start_point + time_steps].reset_index(drop=True)
        #     y_sim = y[start_point:start_point + time_steps].reset_index(drop=True)
        #     # y_sim.reset_index(drop=True)
        #     print(y_sim)
        #     # Set initial conditions for the simulation
        #     m.time = np.linspace(0, time_steps-1, time_steps)
        #     uc[0].value = u_sim[input1]
        #     yc[0].value = y_sim[output1][0]
            
        #     # Solve the model for simulation
        #     # m.options.TIME_SHIFT=0
        #     m.options.IMODE = 4
        #     m.solve(disp=False)
            
        #     # Get the simulated output 
        #     ypred = np.array(yc).T
        #     # adding the values only of array to the list using extend
        #     # ypred_forplot.extend(ypred)
        #     # u_sim_forplot.extend(u_sim[input1])
        #     # t_sim_forplot.extend(np.linspace(start_point, start_point+time_steps-1,time_steps))
        #     # print(t_sim_forplot,u_sim_forplot)
        #     # getting MSE
        #     mse = np.mean((ypred - y_sim)**2, axis=0)
        #     mse_result.append(mse)
        #     # print("Total error (MSE):", mse)
        
        # totalmse = sum(mse_result)/len(mse_result)
        
        # mse_dict = {'na': [na],
        #             'nb': [nb],
        #             'MSE_depthrate':[totalmse[0]]
        #             }
        # print(mse_dict)
        # mse_data = pd.DataFrame(mse_dict)
        # mse_data.to_csv('mse_values_test.csv', mode='w' if first==True else 'a', header=(True if first==True else False), index=False)
        # first = False

        # titlesize = 22
        # fontsize = 12
        # # uc[1].value = u[input2]
        # # Plot the results
        # plt.subplot(2, 1, 1)
        # plt.title('ARX Model Validation', fontsize=titlesize)
        # plt.plot(m.time, uc[0], 'b-', label='BE')
        # # plt.plot(m.time, uc[1], 'r-', label='MM scaled by 20')
        # plt.ylabel('Actuator Value', fontsize=fontsize)
        # plt.xticks(fontsize=fontsize)
        # plt.yticks(fontsize=fontsize)
        # plt.legend(bbox_to_anchor=(1.05, 1.0), ncol=1,
        #            loc="upper left", fontsize=fontsize)

        # print(yc)
        # plt.subplot(2, 1, 2)
        # plt.plot(m.time, y[output1], 'b-',
        #          label='Depth Rate Measured (mm/s)')
        # plt.plot(m.time, ypred, 'g:',
        #          label=r'Depth Rate Predicted Test')
        # # plt.plot(m.time, yc[0], 'g:',
        # #          label=r'Depth Rate Predicted Test', linewidth=2)

       
        # plt.ylabel('Output', fontsize=fontsize)
        # plt.xlabel('Time(s)', fontsize=fontsize)
        # plt.legend(bbox_to_anchor=(1.05, 1.0), ncol=1,
        #            loc="upper left", fontsize=fontsize)
        # plt.xticks(fontsize=fontsize)
        # plt.yticks(fontsize=fontsize)
        # plt.tight_layout()
        # plt.savefig("ValidationData.png")
        # plt.show()

