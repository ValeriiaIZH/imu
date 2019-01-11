import numpy as np
from matplotlib import pyplot as plt
from math import *

d_time = 0.01
array = []
accel = []
ax = []
ay = []
az = []
gx = []
gy = []
gz = []
vel = []
yaw = []
pitch = []
roll = []
v = 0

with open("dt 0_01.txt", 'r') as file:
    data = file.readlines()
    for line in data:
        array.append([float(x) for x in line.split()])
    num = sum(1 for line in data)
    for i in range(0, num):
        gx.append(array[i][1])
        gy.append(array[i][2])
        gz.append(array[i][3])
        ax.append(array[i][4]*(-1))
        ay.append(array[i][5]*(-1))
        az.append(array[i][6])
        # yaw.append(atan(gz[i]/sqrt(gx[i]*gx[i] + gz[i]*gz[i])))
        yaw.append(array[i][8] * 3.14159  / 180)
        pitch.append(array[i][9] * 3.14159 / 180)
        roll.append(array[i][10] * 3.14159 / 180)
        accel.append(np.sqrt(array[i][4] ** 2 + array[i][5] ** 2 + array[i][6] ** 2))

'''plt.plot(ax)
plt.title("Accelaration in X-direction(m/s2)")
plt.xlabel("time step")
plt.ylabel("Accelaration")
plt.show()

plt.plot(ay)
plt.title("Accelaration in Y-direction(m/s2)")
plt.xlabel("time step")
plt.ylabel("Accelaration")
plt.show()

plt.plot(az)
plt.title("Accelaration in Z-direction(m/s2)")
plt.xlabel("time step")
plt.ylabel("Accelaration")
plt.show()
'''
for i in range(0, num):
    v = v + ax[i]*d_time
    vel.append(v)
vel = np.array(vel)

'''
plt.plot(vel)
plt.title("Velocity in X-direction(m/s)")
plt.xlabel("time step")
plt.ylabel("Velocity")
plt.show()

plt.plot(ax)
plt.title("Length of accel vector")
plt.xlabel("time step")
plt.ylabel("Length")
plt.show()
'''
dy = vel * d_time
'''
plt.plot(dy)
plt.title("Displacement")
plt.xlabel("time step")
plt.ylabel("Displacement")
plt.show()
'''
dist = []
d = 0
for i in range(0, num):
    d = d + dy[i]
    dist.append(d)
'''
plt.plot(dist)
plt.title("Total Distance in meter")
plt.xlabel("time step")
plt.ylabel("Distance")
plt.show()
'''
rot_angle = 2 * np.arcsin(yaw)
'''
plt.plot(rot_angle)
plt.title("Rotational Vector in radian")
plt.xlabel("time step")
plt.ylabel("angle")
plt.show()
'''
x = 0
y = 0
x_arr = []
y_arr = []
for i in range(0, num-1):
    x = x + dy[i+1] * np.sin(rot_angle[i])
    y = y + dy[i+1] * np.cos(rot_angle[i])
    x_arr.append(x)
    y_arr.append(y)

plt.scatter(x_arr, y_arr)
plt.title("Trajectory ")
plt.xlabel("y-axis")
plt.ylabel("x-axis")
plt.show()

