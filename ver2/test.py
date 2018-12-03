import math
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from mpl_toolkits.mplot3d import Axes3D


filename = "data.txt"
array = []
time = []
accel = []
gyro = []
magn = []
steps_array = []
velocity = []
d_time = 0.01
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


# lambda z, x, y: z + x * (y if y is not None else 1)

def func(current, previous, factor):
    return previous + (current * factor)

with open(filename, 'r') as file:
    data = file.readlines()
    for line in data:
        array.append([float(x) for x in line.split()])
    num = sum(1 for line in data)
    # print (num)
    for i in range(0, num):
        accel.append(math.sqrt(array[i][3]**2 + array[i][4]**2 + array[i][5]**2))
        # gyro = [array[i][0], array[i][1], array[i][2]]
        # magn = [array[i][6], array[i][7], array[i][8]]
    # print(accel)

    for i in range(0, num):
        if abs(accel[i]) < 1.5:
            static_array = accel[i]
        # print(static_array)

    for i in range(0, num):
        acc_x = array[i][3]
        acc_y = array[i][4]
        acc_z = array[i][5] - 9.81
    # print(acc_z)

        velocity_x.append(acc_x * d_time)
        velocity_y.append(acc_y * d_time)
        velocity_z.append(acc_z * d_time)

        position_x = [i * d_time for i in velocity_x]
        position_y = [i * d_time for i in velocity_y]
        position_z = [i * d_time for i in velocity_z]
    #print(velocity_x)
    #print(position_x)
    for n in range(1, num-1):
        vel_x.append(velocity_x[n-1] + velocity_x[n])
        vel_y.append(velocity_y[n-1] + velocity_y[n])
        vel_z.append(velocity_z[n-1] + velocity_z[n])

        pos_x.append(position_x[n-1] + position_x[n])
        pos_y.append(position_y[n-1] + position_y[n])
        pos_z.append(position_z[n-1] + position_z[n])
    #print(vel_x)

with open("time_d.txt", "r") as file_time:
    time_d = file_time.readlines()
    for line in time_d:
        time.append([float(t) for t in line.split()])
    # print(time)

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
X = pos_x
Y = pos_y
Z = pos_z
t = time
ax.plot(X, Y, Z)


plt.show()






