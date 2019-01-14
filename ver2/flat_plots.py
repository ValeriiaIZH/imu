import numpy as np
from matplotlib import pyplot as plt
from math import *

d_time = 0.1
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

with open("sportzal4.txt", 'r') as file:
    data = file.readlines()
    for line in data:
        array.append([float(x) for x in line.split()])
    num = sum(1 for line in data)
    for i in range(0, num):

        ax.append(array[i][0])
        ay.append(array[i][1])
        az.append(array[i][2])

        gx.append(array[i][3])
        gy.append(array[i][4])
        gz.append(array[i][5])
        yaw.append(atan(gz[i]/sqrt(gx[i]*gx[i] + gz[i]*gz[i])))
        pitch.append( atan(ax[i] / sqrt(ay[i] * ay[i] + az[i]* az[i])))
        roll.append( atan(ay[i]/ sqrt(ax[i] * ax[i]+ az[i] * az[i])))
        #yaw.append(array[i][9] * pi / 180)
        #pitch.append(array[i][10] * pi / 180)
        #roll.append(array[i][11] * pi / 180)
        accel.append(np.sqrt(array[i][4] ** 2 + array[i][5] ** 2 + array[i][6] ** 2))

vel = []
v = 0
for i in range(0, num):
    v = v + ay[i]*d_time
    vel.append(v)
vel = np.array(vel)

dy = vel * d_time

dist = []
d = 0
for i in range(0, num):
    d = d + dy[i]
    dist.append(d)

rot_angle = 2 * np.arcsin(yaw)
#rot_angle = 2 * np.arcsin(pitch)
#rot_angle = 2 * np.arcsin(roll)

x = 0
y = 0
x_arr = []
y_arr = []
for i in range(0, num-1):
    x = x + dy[i+1] * np.sin(rot_angle[i])
    y = y + dy[i+1] * np.cos(rot_angle[i])
    x_arr.append(x)
    y_arr.append(y)


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

plt.plot(dy)
plt.title("Displacement")
plt.grid()
plt.xlabel("time step")
plt.ylabel("Displacement")
plt.show()

plt.plot(dist)
plt.title("Total distance in meter")
plt.grid()
plt.xlabel("time step")
plt.ylabel("distance")
plt.show()

plt.plot(rot_angle)
plt.title("Rotational vector in radian")
plt.grid()
plt.xlabel("time step")
plt.ylabel("angle")
plt.show()

plt.scatter(x_arr, y_arr )
plt.title("Trajectory ")
plt.xlabel("x-axis")
plt.ylabel("y-axis")
plt.show()
