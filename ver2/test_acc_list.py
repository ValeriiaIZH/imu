'''
Python code which represents 3D trajectory from IMU(blue) and real trajectory(red)
There aren't functions in this code
Only lists were used
File data_from_gym2.txt consist of values from accelerometr, gyroscope, magnetometr
and angles in such form:
accel_x  accel_y   accel_z  gyro_x  gyro_y  gyro_z  magn_x  magn_y  magn_z  yaw   pitch  roll
'''

from math import *
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from mpl_toolkits.mplot3d import Axes3D

array = []
accel = []
yaw = []
pitch = []
roll = []
velocity = []
static = []
dist = []
v = 0
d = 0
x = 0
y = 0
z = 0
d_t = 0.0025 #time step

filename = "data_from_gym2.txt"

# read data from file
with open(filename, 'r') as file:
    data = file.readlines()
    for line in data:
        array.append([float(x) for x in line.split()])
    num = sum(1 for line in data)
    # print (num)
    for i in range(0, num):
        accel.append(np.sqrt(array[i][1]**2 + array[i][2]**2 + array[i][3]**2)) # vector length of accel

        yaw.append(array[i][9] * pi / 180)  # from degrees to radians
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
    # print(accel)
    # calculate velocity from vector length of accel
    for i in range(0, num):
        v = v + accel[i] * d_t
        velocity.append(v)
    velocity = np.array(velocity)
    dy = velocity * d_t

    for i in range(0, num):
        d = d + dy[i]
        dist.append(d)
    rot_angle_yaw = 2 * np.arcsin(yaw) * np.arcsin(pitch) * np.arcsin(roll)
    # rot_angle_pitch = 2 * np.arcsin(pitch)
    # rot_angle_roll = 2 * np.arcsin(roll)

# calculate points for trajectory
x_arr = []
y_arr = []
z_arr = []
for i in range(0, num - 1):
    x = x + dy[i + 1] * np.sin(rot_angle_yaw[i])
    y = y + dy[i + 1] * np.cos(rot_angle_yaw[i])
    z = z + dy[i + 1]
    x_arr.append(x)
    y_arr.append(y)
    z_arr.append(z)

# real trajectory
real_coordinate_x = [0, 2.5, 2.5, 10]
real_coordinate_y = [0, 0, 5, 6]

# plot figures
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot(x_arr, y_arr,  color = 'blue')
ax.plot(real_coordinate_x, real_coordinate_y, 'r')
ax.set_xlabel('X Label')
ax.set_ylabel('Y Label')
ax.set_zlabel('Z Label')
plt.show()
