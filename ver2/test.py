import math
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from mpl_toolkits.mplot3d import Axes3D


filename = "imu_data.txt"
d_time = 0.02
array = []
static_array = []

m_array = {'time': [], 'accel': [], 'gyro': [], 'magn': []}
velocity = {'x': [], 'y': [], 'z': []}
position = {'x': [], 'y': [], 'z': []}
m_position = {'x': [], 'y': [], 'z': []}

with open(filename, 'r') as file:
    data = file.readlines()
    for line in data:
        array.append([float(x) for x in line.split()])
    num = sum(1 for line in data)
    # print (num)
    for i in range(0, num):
        m_array['accel'].append(np.sqrt(array[i][1]**2 + array[i][2]**2 + array[i][3]**2))
        # m_array['gyro'].append([array[i][4], array[i][5], array[i][6]])
    # print(m_array['accel'])

    for i in range(0, num):
        if abs(m_array['accel'][i]) < 1.5:
            static_array.append(m_array['accel'][i])
    # print(static_array)

    for k in range(0, num):
        acc_x = array[k][3]
        acc_y = array[k][4]
        acc_z = array[k][5] - 9.81
        # print(acc_z)

        velocity['x'].append(acc_x * d_time)
        velocity['y'].append(acc_y * d_time)
        velocity['z'].append(acc_z * d_time)

        position['x'] = [i * d_time for i in velocity['x']]
        position['y'] = [i * d_time for i in velocity['y']]
        position['z'] = [i * d_time for i in velocity['z']]
    # print(velocity['x'])
    # print(position['x'])

    for n in range(1, num - 1):
        velocity['x'][n] = velocity['x'][n - 1] + velocity['x'][n]
        velocity['y'][n] = velocity['y'][n - 1] + velocity['y'][n]
        velocity['z'][n] = velocity['z'][n - 1] + velocity['z'][n]

        m_position['x'].append(position['x'][n - 1] + position['x'][n])
        m_position['y'].append(position['y'][n - 1] + position['y'][n])
        m_position['z'].append(position['z'][n - 1] + position['z'][n])
    # print(velocity['x'])
    # print(m_position['x'])





