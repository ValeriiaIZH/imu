import math
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from mpl_toolkits.mplot3d import Axes3D


filename = "imu_data.txt"
d_time = 0.02
array = []
static_array = []
time = []
accel = []
gyro = []
magn = []
velocity = []
velocity_x = []
vel_x = []
velocity_y = []
vel_y = []
velocity_z = []
vel_z = []
position_x = []
pos_x = []
position_y = []
pos_y = []
position_z = []
pos_z = []

with open(filename, 'r') as file:
    data = file.readlines()
    for line in data:
        array.append([float(x) for x in line.split()])
    num = sum(1 for line in data)
    # print (num)
    for i in range(0, num):
        accel.append(np.sqrt(array[i][1]**2 + array[i][2]**2 + array[i][2]**2))
        gyro = [array[i][4], array[i][5], array[i][6]]
    # print(m_array['accel'])


    for i in range(0, num):
        if abs(accel[i]) < 1.5:
            static_array.append(accel[i])
    # print(static_array)

    for i in range(0, num):
        acc_x = array[i][1]
        acc_y = array[i][2]
        acc_z = array[i][3] - 9.81
    # print(acc_z)

        velocity_x.append(acc_x * d_time)
        velocity_y.append(acc_y * d_time)
        velocity_z.append(acc_z * d_time)

        position_x = [i * d_time for i in velocity_x]
        position_y = [i * d_time for i in velocity_y]
        position_z = [i * d_time for i in velocity_z]
    # print(velocity_x)
    # print(position_x)
    for n in range(1, num-1):
        pos_x.append(position_x[n-1] + position_x[n])
        pos_y.append(position_y[n-1] + position_y[n])
        pos_z.append(position_z[n-1] + position_z[n])
    # print(pos_x)

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
# ax = plt.subplot()
ax.plot(pos_x, pos_y, pos_z, color = 'blue')
plt.show()

