from math import *
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from mpl_toolkits.mplot3d import Axes3D

array = []
gyro = []
magn = []
accel = []
mad_x = []
mag_y = []
mag_z =[]
yaw = []
pitch = []
roll = []
velocity = []
v = 0
dist = []
d = 0
x = 0
y = 0
z = 0
x_arr = []
y_arr = []
z_arr = []
d_t = 0.0025
filename = "data_from_gym2.txt"

with open(filename, 'r') as file:
    data = file.readlines()
    for line in data:
        array.append([float(x) for x in line.split()])
    num = sum(1 for line in data)
    # print (num)
    for i in range(0, num):
        accel.append(np.sqrt(array[i][0]**2 + array[i][1]**2 + array[i][2]**2))

        yaw.append(array[i][9] * pi / 180)
        pitch.append(array[i][10] * pi / 180)
        roll.append(array[i][11] * pi / 180)
        if yaw[i] > 1:
            yaw[i] = 1
        if yaw[i] < -1:
            yaw[i] = -1
        if pitch[i] > 1:
            pitch[i] = 1
        if pitch[i] < -1:
            pitch[i] = -1
        if roll[i] > 1:
            roll[i] = 1
        if roll[i] < -1:
            roll[i] = -1
        #gyro.append(array[i][3], array[i][4], array[i][5])
        #magn.append(array[i][6], array[i][7], array[i][8])
    # print(accel)
    for i in range(0, num):
        v = v + accel[i] * d_t
        velocity.append(v)
    velocity = np.array(velocity)
    dy = velocity * d_t

    for i in range(0, num):
        d = d + dy[i]
        dist.append(d)
    rot_angle_yaw = 2 * np.arcsin(yaw) * np.arcsin(pitch) * np.arcsin(roll)
    #rot_angle_pitch = 2 * np.arcsin(pitch)
    #rot_angle_roll = 2 * np.arcsin(roll)

    for i in range(0, num - 1):
        x = x + dy[i + 1] * np.sin(rot_angle_yaw[i])
        y = y + dy[i + 1] *np.cos(rot_angle_yaw[i]) #* np.sin(rot_angle_pitch[i])
        z = z + dy[i + 1] #*np.cos(rot_angle_yaw[i])#* np.cos(rot_angle_pitch[i])
        x_arr.append(x)
        y_arr.append(y)
        z_arr.append(z)

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
# ax = plt.subplot()
ax.plot(x_arr, y_arr,  color = 'blue')
ax.set_xlabel('X Label')
ax.set_ylabel('Y Label')
ax.set_zlabel('Z Label')
plt.show()