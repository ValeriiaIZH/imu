'''
Python code which represents 3D trajectory from IMU(blue) and real trajectory(red)
There aren't functions in this code
Only lists were used
File data_from_gym2.txt consist of values from accelerometre, gyroscope, magnetometre
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
gyro_x = []
gyro_y = []
gyro_z = []
yaw = []
pitch = []
roll = []
velocity = []
static = []
dist = []
angle_x = []
angle_y = []
angle_z = []
v = 0
d = 0
x = 0
y = 0
z = 0
ag_x = 0
ag_y = 0
ag_z = 0
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
        accel.append(np.sqrt(array[i][0]**2 + array[i][1]**2 + array[i][2]**2)) # vector length of accel
        gyro_x.append(array[i][3])
        gyro_y.append(array[i][4])
        gyro_z.append(array[i][5])
        '''yaw.append(array[i][9] * pi / 180)  # from degrees to radians
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
            roll[i] = -1'''
    # print(accel)
    # integrate gyro data
    for i in range(0, num):
        ag_x = ag_x + gyro_x[i] * d_t
        angle_x.append(ag_x)
    angle = np.array(angle_x)

    for i in range(0, num):
        ag_y = ag_y + gyro_y[i] * d_t
        angle_y.append(ag_y)
    angle_y = np.array(angle_y)

    for i in range(0, num):
        ag_z = ag_z + gyro_z[i] * d_t
        angle_z.append(ag_z)
    angle_z = np.array(angle_z)

    # calculate velocity from vector length of accel
    for i in range(0, num):
        v = v + accel[i] * d_t
        velocity.append(v)
    velocity = np.array(velocity)
    dy = velocity * d_t

    for i in range(0, num):
        d = d + dy[i]
        dist.append(d)
    rot_angle_x = 2 * np.arcsin(angle_x)
    rot_angle_y = 2 * np.arcsin(angle_y)
    rot_angle_z = 2 * np.arcsin(angle_z)

# calculate points for trajectory
x_arr = []
y_arr = []
z_arr = []
for i in range(0, num - 1):
    x = x + dy[i + 1] * np.sin(rot_angle_z[i])
    y = y + dy[i + 1] * np.cos(rot_angle_z[i]) * np.cos(rot_angle_y[i])
    z = z + dy[i + 1] * np.sin(rot_angle_y[i]) * np.sin(rot_angle_x[i])
    x_arr.append(x)
    y_arr.append(y)
    z_arr.append(z)

# real trajectory
real_coordinate_x = [0, 0, -4, -4]
real_coordinate_y = [0, 2.5, 3, 10]

# plot figures
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot(x_arr, y_arr, z_arr, color = 'blue')
ax.plot(real_coordinate_x, real_coordinate_y, 'r')
ax.set_xlabel('X Label')
ax.set_ylabel('Y Label')
ax.set_zlabel('Z Label')
plt.show()
