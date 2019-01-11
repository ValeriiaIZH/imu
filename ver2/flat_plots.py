import numpy as np
from matplotlib import pyplot as plt
from math import *

d_time = 0.01
array = []
accel = []
gx = []
gy = []
gz = []
ax = []
ay = []
az = []
yaw = []
pitch = []
roll = []

with open("dt 0_01.txt", 'r') as file:
    data = file.readlines()
    for line in data:
        array.append([float(x) for x in line.split()])
    # num = sum(1 for line in data)
    num = 352
    for i in range(0, num):
        # gx.append(array[i][1])
        # gy.append(array[i][2])
        # gz.append(array[i][3])
        ax.append(array[i][4]*(-1))
        ay.append(array[i][5]*(-1))
        az.append(array[i][6])
        # yaw.append(atan(gz[i]/sqrt(gx[i]*gx[i] + gz[i]*gz[i])))
        yaw.append(array[i][8] * 3.14159 * 0.4/ 180)
        pitch.append(array[i][9] * 3.14159 / 180)
        roll.append(array[i][10] * 3.14159 / 180)
        accel.append(np.sqrt(array[i][4] ** 2 + array[i][5] ** 2 + array[i][6] ** 2))

plt.plot(ax)
plt.title("Accelaration x (m/s2)")
plt.grid()
plt.xlabel("time step")
plt.ylabel("Accelaration")
plt.show()

plt.plot(ay)
plt.title("Accelaration y (m/s2)")
plt.grid()
plt.xlabel("time step")
plt.ylabel("Accelaration")
plt.show()

plt.plot(az)
plt.title("Accelaration z (m/s2)")
plt.grid()
plt.xlabel("time step")
plt.ylabel("Accelaration")
plt.show()

vel = []
v = 0
for i in range(0, num):
    v = v + ax[i]*d_time
    vel.append(v)
vel = np.array(vel)


plt.plot(vel)
plt.title("Velocity in x (m/s)")
plt.grid()
plt.xlabel("time step")
plt.ylabel("Velocity")
plt.show()

plt.plot(ax)
plt.title("Length of accel vector")
plt.grid()
plt.xlabel("time step")
plt.ylabel("Length")
plt.show()

dy = vel * d_time

plt.plot(dy)
plt.title("Displacement")
plt.grid()
plt.xlabel("time step")
plt.ylabel("Displacement")
plt.show()

dist = []
d = 0
for i in range(0, num):
    d = d + dy[i]
    dist.append(d)

plt.plot(dist)
plt.title("Total distance in meter")
plt.grid()
plt.xlabel("time step")
plt.ylabel("distance")
plt.show()

rot_angle = 2 * np.arcsin(yaw)

plt.plot(rot_angle)
plt.title("Rotational vector in radian")
plt.grid()
plt.xlabel("time step")
plt.ylabel("angle")
plt.show()

x = 0
y = 0
x_arr = []
y_arr = []
for i in range(0, num-1):
    x = x + dy[i+1] * np.sin(rot_angle[i])
    y = y + dy[i+1] * np.cos(rot_angle[i])
    x_arr.append(x)
    y_arr.append(y)

plt.scatter(y_arr, x_arr)
plt.title("Trajectory ")
plt.xlabel("x-axis")
plt.ylabel("y-axis")
plt.show()

