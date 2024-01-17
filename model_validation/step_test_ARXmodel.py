from gekko import GEKKO
import numpy as np
import matplotlib.pyplot as plt
import pickle

# with open('p_values.pickle', 'rb') as file:
#     p = pickle.load(file)
# with open('p_mmpitchval.pickle', 'rb') as file:
#     p_mm_pitch = pickle.load(file)
    
with open('p_br_dr.pickle', 'rb') as file:
    p = pickle.load(file)
    
# with open('p_berate_depthrate.pickle', 'rb') as file:
#     p_br_pitch = pickle.load(file)


print(p)

# p = {'a': np.array([[ p_br_dr['a'][0][0], p_mm_pitch['a'][0][0]],
#        [ p_br_dr['a'][1][0] , p_mm_pitch['a'][1][0]],
#        [-0.00,0]]), 'b': np.array([[[ p_br_dr['b'][0][0][0],  9.15255674e-09, 0.0],
#                                          [0, 0, 0]],

#        [[p_br_pitch['b'][0][0][0],  0.00702477, -0.00871537],
#         [0, 0, 0]
#         ]]), 'c': np.array([ 0, -0.05889666])}

# p = {'a': np.array([[ 1,  1.79525867],
#        [ 0, -0.82545468]]), 'b': np.array([[[-0.03728023, -0.00000009422761],
#                                             [0.14529229,0],
#                                             [0.0775675,0],
#                                             [0.09086155,0],
#                                             [-0.00651081,0]],
#        [[-0.00262495,  0.00183597],
#          [0,0],
#         [0,0],
#         [0.09086155,0],
#         [-0.00651081,0]]
#         ]), 'c': np.array([0., 0.])}

# Create GEKKO model
m = GEKKO(remote=False)

yc, uc = m.arx(p)
m.options.IMODE = 1
m.solve(disp=False)

step = 400
m.time = np.arange(0,step,1)
# m.time = np.linspace(0, step, step/2)

uc[0].value = np.zeros(len(m.time))
# uc[1].value = np.zeros(len(m.time))
maxValueBErate = 20
ramptime = 20
# for i in range(ramptime):
#     step = maxValueBErate/ramptime
#     uc[0].value[i] = step*i + 0
uc[0].value[ramptime:] = maxValueBErate
uc[0].value[ramptime+20:] = 0
# adding the 
# for i in range(50):
#     step = 250/50
#     uc[1].value[i] = step*i + 0
# uc[1].value[50:] = 250
# for i in range(200):
#     step = -500/200
#     uc[1].value[200+i] = step*i + 250
# uc[1].value[400:] = -250
# for i in range(200):
#     step = -700/100
#     uc[0].value[200+i] = step*i + 350
# uc[0].value[300:] = -350
# uc[1].value = 0
yc[0].value[0] = 0
# yc[1].value[0] = 0
m.options.IMODE = 4
m.options.NODES=2
m.solve(disp=False)

plt.suptitle('Ramp Test')
plt.subplot(2, 2, 1)
plt.title('BE Ramp Input')
plt.plot(m.time, uc[0].value, 'b-', label='BE')
# plt.plot(m.time, uc[1].value, 'r-', label='MM')
plt.ylabel('Input Value')
plt.legend()
plt.subplot(2, 2, 3)
plt.plot(m.time, yc[0].value, 'b--', label='Depth_rate')
# plt.plot(m.time, yc[1].value, 'r--', label='Pitch')
plt.ylabel('Output Value')
plt.xlabel('Time (sec)')
plt.legend()

# # step for second MV (Heater 2)
# uc[0].value = np.zeros(len(m.time))
# uc[1].value = np.zeros(len(m.time))
# uc[0].value[:] = 0
# maxValueMM = 200
# ramptime = 100
# for i in range(ramptime):
#     step = maxValueMM/ramptime
#     uc[1].value[i] = step*i + 0
# uc[1].value[ramptime:] = step*ramptime
# minValueMM = -200
# ramptime = 200
# lagtime = 200
# for i in range(ramptime):
#     step = minValueMM * 2/ramptime 
#     uc[1].value[lagtime+i] = step*i + maxValueMM
# uc[1].value[lagtime+ramptime:] = -200
# yc[0].value[0] = 0
# yc[1].value[0] = 0
# m.solve(disp=False)
# plt.subplot(2, 2, 2)

# plt.title('MM Ramp Input')
# plt.plot(m.time, uc[0].value, 'b-', label='BE')
# plt.plot(m.time, uc[1].value, 'r-', label='MM')
# plt.ylabel('Input Value')
# plt.legend()
# plt.subplot(2, 2, 4)

# plt.plot(m.time, yc[0].value, 'b--', label='Depth_rate')
# plt.plot(m.time, yc[1].value, 'r--', label='Pitch')
# plt.ylabel('Output Value')
# plt.xlabel('Time (sec)')
# plt.legend()
# plt.tight_layout()
# plt.savefig('step_test.png', dpi=300)
plt.show()
