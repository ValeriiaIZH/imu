'''
NOT READY YET
Pure code with functions
'''
import math
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from mpl_toolkits.mplot3d import Axes3D

d_time = 0.02


def main():
    m_acc, m_gyro, acceleration_x, acceleration_y, acceleration_z, num_of_strings = read_data("imu_data.txt")
    #filtred_array = filter(m_acc, num_of_strings)
    abs_acc = np.abs(m_acc)
    step_array = srez_step(m_acc, num_of_strings)
    m_pos = {'x': [], 'y': [], 'z': []}
    m_pos = count_vel_and_pos(m_acc, acceleration_x, acceleration_y, acceleration_z, d_time, num_of_strings)
    print(m_pos)


def read_data(filename):
    array = []
    m_array = {'time': [], 'acc': [], 'gyro': [], 'magn': []}
    acc_x = []
    acc_y = []
    acc_z = []
    with open(filename, 'r') as file:
        data = file.readlines()
        for line in data:
            array.append([float(x) for x in line.split()])
        num = sum(1 for line in data)
        # print (num)
        for i in range(0, num):
            m_array['acc'].append(np.sqrt(array[i][1] ** 2 + array[i][2] ** 2 + array[i][3] ** 2))
            # m_array['gyro'].append([array[i][4], array[i][5], array[i][6]])
            acc_x.append(array[i][1])
            acc_y.append(array[i][2])
            acc_z.append(array[i][3] - 9.81)
        print(m_array['acc'])
        print(acc_x)
    return m_array['acc'], m_array['gyro'], acc_x, acc_y, acc_z, num

'''
# band-pass filter (a merge of low pass and high pass filters
def filter(array, num):
    f_l = 0.1
    f_h = 0.3
    b = 0.08
    n = int(np.ceil(4/b))
    if not n % 2: n += 1
    n = np.arange(n)

    # low pass filter
    hlpf = np.sinc(2 * f_h * (n - (n - 1) / 2.))
    hlpf *= np.blackman(n)
    hlpf = hlpf / np.sum(hlpf)

    # high pass filter
    hhpf = np.sinc(2 * f_l * (n - (n - 1) / 2.))
    hhpf *= np.blackman(n)
    hhpf = hhpf / np.sum(hhpf)
    hhpf = -hhpf
    hhpf[int((n -1) / 2)] += 1

    h = np.convolve(hlpf, hhpf)
    s = array
    filtred_arr = np.convolve(s, h)

    return filtred_arr
'''

def srez_step(array, num):
    static_array = []
    for i in range(0, num):
        if array[i] < 1.5:
            static_array.append(array[i])
    return static_array


def count_vel_and_pos(array, acc_x, acc_y, acc_z, time, num):
    velocity = {'x': [], 'y': [], 'z': []}
    position = {'x': [], 'y': [], 'z': []}
    m_position = {'x': [], 'y': [], 'z': []}

    for k in range(0, num):
        velocity['x'].append(acc_x * d_time)
        velocity['y'].append(acc_y * d_time)
        velocity['z'].append(acc_z * d_time)

        position['x'] = [i * d_time for i in velocity['x']]
        position['y'] = [i * d_time for i in velocity['y']]
        position['z'] = [i * d_time for i in velocity['z']]

    for n in range(1, num - 1):
        m_position['x'].append(position['x'][n - 1] + position['x'][n])
        m_position['y'].append(position['y'][n - 1] + position['y'][n])
        m_position['z'].append(position['z'][n - 1] + position['z'][n])
        # print(m_position)
    return m_position

"""
def plots(m_position):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    x = m_position['x']
    y = m_position['y']
    z = m_position['z']
    ax.plot(x, y, z, color='blue')
    plt.show()
"""


if __name__ == '__main__':
    main()
