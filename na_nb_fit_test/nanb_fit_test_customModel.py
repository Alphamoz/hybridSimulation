from gekko import GEKKO
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pickle
first = True
na_s = [1,2,3,4,5,6]  # output coefficients
nb_s = [1,2,3,4,5,6]  # input coefficients

# function to load the data
def initializeData(data, inputs, outputs, time, startData, endData, normalization = False, scaleOutput = False):
    data = pd.read_csv(data)
    # Model identification MM to Pitch
    t = data[time]
    u = data.loc[:,inputs]
    y = data.loc[:,outputs]
    u = u[startData:].reset_index(drop=True) if endData == 0 else u[startData:endData].reset_index(drop=True)
    y = y[startData:].reset_index(drop=True) if endData == 0 else y[startData:endData].reset_index(drop=True)
    t = t[0:u.shape[0]]
    if normalization:
        u[inputs[0]][:] -= normalization
    if scaleOutput:
        y[outputs[0]][:] *= scaleOutput
    return u, y, t

for na in na_s:
    for nb in nb_s:        
        # mm, pitch model
        u,y,t = initializeData(data='./data/dataProcessing/processedData/DataModel5detik.csv',inputs = ["MM"], outputs = ["Pitch"],time=["Time"], startData=150, endData=0)
        m = GEKKO(remote=False)
        yp_mm_pitch, p_mm_pitch, K_mm_pitch = m.sysid(t, u=u,
                   y=y, na=na, nb=nb, pred='meas', shift='none', scale=False)
        # # Calculate mean squared error to determine best na and nb
        mse_pitch = np.mean((yp_mm_pitch - y)**2, axis=0)
        print(yp_mm_pitch - y)
        # be, depth rate model
        u,y,t = initializeData(data='./data/dataProcessing/processedData/DataModel5detik.csv',inputs = ["be_rate"], outputs = ["depth_rate"], time=["Time"],startData=2, endData=40, normalization=0, scaleOutput=0)
        m = GEKKO(remote=False)
        yp_be_dr, p_be_dr, K_be_dr = m.sysid(t, u=u,
                   y=y, na=na, nb=nb, pred='meas', shift='none', scale=False)
        mse_be_dr = np.mean((yp_be_dr - y)**2, axis=0)
        print(yp_be_dr - y)
        # be, pitch model
        u,y,t = initializeData(data='./data/dataProcessing/processedData/DataModel5detik.csv',inputs = ["be_rate"], outputs = ["Pitch"], time=["Time"], startData=2, endData=40, normalization=0)
        m = GEKKO(remote=False)
        yp_be_pitch, p_be_pitch, K_be_pitch = m.sysid(t, u=u,
                   y=y, na=na, nb=nb, pred='meas', shift='none', scale=False)
        # # Calculate mean squared error to determine best na and nb
        mse_be_pitch = np.mean((yp_be_pitch - y)**2, axis=0)
        print(mse_be_pitch)
        # # Print the total error
        print("Total error (MSE):", mse_be_dr["depth_rate"]+mse_pitch["Pitch"]+mse_be_pitch['Pitch'])
        mse_dict = {'na': [na],
                    'nb': [nb],
                    'MSE_depthrate':[mse_be_dr["depth_rate"]],
                    'MSE_pitch': [mse_pitch["Pitch"]+mse_be_pitch['Pitch']]
                    }
        print(mse_dict)
        mse_data = pd.DataFrame(mse_dict)
        mse_data.to_csv('mse_values_train.csv', mode='w' if first==True else 'a', header=(True if first==True else False), index=False)


        # data validation
        data = pd.read_csv('./data/dataProcessing/processedData/downsampled_data.csv')
        input1 = "field.internal_volume"
        input2 = "field.cur_mm"
        output1 = "field.depth_rate"
        output2 = "field.pitch_data"
        t = data['Time']
        u = data.loc[:,[input1, input2]]
        y = data.loc[:,[output1, output2]]
        startData = 15
        endData = 25
        u = u[startData:endData].reset_index(drop=True)
        y = y[startData:endData].reset_index(drop=True)
        t = t[0:u.shape[0]]
        u[input1] -=350
        u[input2] -= 250
        y[output1] *= 1000

        # m = GEKKO(remote=False)
        
        p_a = np.array([])
        for i in range(na):
            p_a = np.append(p_a, [ p_be_dr['a'][i][0], p__pitch['a'][i][0]])
        p_a = p_a.reshape(-1, 2)
        
        # print(p_a)
        p_b = np.array([])
        # b for depthrate
        for i in range(nb):
            p_b = np.append(p_b, [ p_be_dr['b'][0][i][0], 0])
        # p_b = p_b.reshape(1,-1, 2)
        for i in range(nb):
            p_b = np.append(p_b, [p_be_pitch['b'][0][i][0], p_mm_pitch['b'][0][i][0]])
        p_b = p_b.reshape(2,-1, 2)
        # print(p_b)
        p_c = np.array([])
        p_c = np.append(p_c, [p_be_dr['c'][0], p_be_pitch['c'][0]])
            
            
        p={'a' : p_a, 'b' : p_b, 'c' : p_c}

        yc, uc = m.arx(p)
        m.options.IMODE = 1
        m.solve(disp=False)

        m.time = np.linspace(0, u.shape[0]-1, u.shape[0])
        # simulation
        # m.options.TIME_SHIFT=0
        m.options.IMODE = 4
        uc[0].value = u[input1]
        uc[1].value = u[input2]
        yc[0].value = y[output1][0]
        yc[1].value = y[output2][0] 
        print(y[output1][0])
        print(y[output2][0])
        # print(yc[0].value[0])
        # yc[0].value = y[output1][0:3]
        # yc[1].value = y[output2][0:3]

        m.solve(disp=False)

        print(type(y))
        # print(type(yc))
        yc = np.array(yc).T
        # print(yc_array-y)
        mse = np.mean((yc - y)**2, axis=0)

        print("Total error (MSE):", mse)
        print(mse)
        # mse_data = pd.DataFrame([mse[0],mse[1]])  # Creating a DataFrame for new values
        mse_dict = {'na': [na],
                    'nb': [nb],
                    'MSE_depthrate':[mse[0]],
                    'MSE_pitch': [mse[1]]
                    }
        print(mse_dict)
        mse_data = pd.DataFrame(mse_dict)
        mse_data.to_csv('mse_values_test.csv', mode='w' if first==True else 'a', header=(True if first==True else False), index=False)
        first = False

        # titlesize = 22
        # fontsize = 12
        # uc[1].value = u[input2]
        # # Plot the results
        # plt.subplot(2, 1, 1)
        # plt.title('ARX Model Validation', fontsize=titlesize)
        # plt.plot(m.time, uc[0], 'b-', label='BE')
        # plt.plot(m.time, uc[1], 'r-', label='MM scaled by 20')
        # plt.ylabel('Actuator Value', fontsize=fontsize)
        # plt.xticks(fontsize=fontsize)
        # plt.yticks(fontsize=fontsize)
        # plt.legend(bbox_to_anchor=(1.05, 1.0), ncol=1,
        #            loc="upper left", fontsize=fontsize)

        # print(yc)
        # plt.subplot(2, 1, 2)
        # plt.plot(m.time, y[output1], 'b-',
        #          label='Depth Rate Measured (mm/s)')
        # plt.plot(m.time, yc[:,:1], 'g:',
        #          label=r'Depth Rate Predicted Test')
        # # plt.plot(m.time, yc[0], 'g:',
        # #          label=r'Depth Rate Predicted Test', linewidth=2)

        # plt.plot(m.time, y[output2], 'c-', label='Pitch Measured (degree)')
        # plt.plot(m.time, yc[:,1:2], 'r:', label=r'Pitch Predicted Test')
        # # plt.plot(m.time, yc[1], 'r:',
        # #          label=r'$Pitch Predicted Test$', linewidth=2)
        # plt.ylabel('Output', fontsize=fontsize)
        # plt.xlabel('Time(s)', fontsize=fontsize)
        # plt.legend(bbox_to_anchor=(1.05, 1.0), ncol=1,
        #            loc="upper left", fontsize=fontsize)
        # plt.xticks(fontsize=fontsize)
        # plt.yticks(fontsize=fontsize)
        # plt.tight_layout()
        # plt.savefig("ValidationData.png")
        # plt.show()

