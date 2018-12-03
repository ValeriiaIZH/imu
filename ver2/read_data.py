import numpy as np
import matplotlib.pyplot as plt
import math


def main():
    m_accel, m_gyro, m_magn, num_of_strings = read_data("data.txt")
    filtred_array = filter(m_accel, num_of_strings)


def read_data(filename):
    array = []
    accel = []
    gyro = []
    magn = []
    with open(filename, 'r') as file:
        data = file.readlines()
        for line in data:
            array.append([float(x) for x in line.split()])
        num = sum(1 for line in data)
        # print (num)
        for i in range(0, num):
            accel = accel.append(math.sqrt(array[i][3]**2 + array[i][4]**2 + array[i][5]**2))
            # gyro = accel.append(math.sqrt(array[i][0]**2 + array[i][1]**2 + array[i][2]**2))
            # magn = accel.append(math.sqrt(array[i][6]**2 + array[i][7]**2 + array[i][8]**2))
            # print(accel)
    return accel, gyro, magn, num


def abs(array):
    abs(array)
    return array

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


def srez_step(array, num):
    static_array = []
    for i in range(0, num):
        if array[i] < 1.5:
            static_array = array[i]
        print(static_array)
    return static_array


def count_vel_and_pos(array, time, num):
    velocity_x = []
    velocity_y = []
    velocity_z = []
    position_x = []
    position_y = []
    position_z = []
    for i in range(0, num):
        acc_x = array[i][3]
        acc_y = array[i][4]
        acc_z = array[i][5] - 9.81
        # print(acc_z)

        velocity_x.append(acc_x * time)
        velocity_y.append(acc_y * time)
        velocity_z.append(acc_z * time)
        position_x = [i * time for i in velocity_x]
        position_y = [i * time for i in velocity_y]
        position_z = [i * time for i in velocity_z]

    return position_x, position_y, position_z, velocity_x, velocity_y, velocity_z


def plots (data):
    fig = plt.figure()
    plt.plot(data, label = r'data')
    plt.xlabel(r'time')
    plt.ylabel(r'data')
    plt.grid(True)
    fig.savefig("plot.png")
    plt.show()


if __name__ == '__main__':
    main()
