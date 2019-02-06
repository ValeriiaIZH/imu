'''
NOT READY YET
Pure code with functions
'''
from math import *
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from mpl_toolkits.mplot3d import Axes3D
from numpy import dot, sum, tile, linalg
from numpy.linalg import inv, pinv
import pandas as pd

d_time = 0.01


class kalman():
    def __init__(self, X, P, A, Q, R, B, U, H):
        self.X = X
        self.P = P
        self.A = A
        self.Q = Q
        self.R = R
        self.B = B
        self.H = H

    def __call__(self):
        pass

    # def predict(X, P, A, Q, B, U):
    def predict(self, U):
        '''
            X - state vector
            P - state covariance matrix
            A - transition matrix
            Q - process covariance matrix
            B - input effect matrix
            U - control input
        '''

        self.X = dot(self.A, self.X) + dot(self.B, U)
        self.P = dot(self.A, dot(self.P, self.A.T)) + self.Q
        return self.X

    # def update(X, P, Y, H, R):
    def update(self, Y):
        '''
            X - state vector
            P - state covariance matrix
            Y - measurement
            H - measurement matrix
            R - measurement covariance matrix
        '''

        Y_pred = dot(self.H, self.X)
        S = dot(self.H, dot(self.P, self.H.T)) + self.R

        if type(S) == np.float64:
            K = dot(self.P, dot(self.H.T, 1 / S))
        else:
            # print S
            K = dot(self.P, dot(self.H.T, pinv(S)))
        # K - Kalman gain
        self.X = self.X + dot(K, (Y - Y_pred))
        self.P = self.P - dot(K, dot(self.H, self.P))
        return self.X


def main():
    # filename = "data_from_gym2.txt"
    filename = "dt 0_01.txt"
    m_accel_vector, m_ax, m_ay, m_az, m_y, m_p, m_r, m_num = read_data(filename)
    hf_acc_vec, lf_acc_vec = lp_and_hp_filter(m_accel_vector, m_num)
    m_steps = detect_step(lf_acc_vec, m_num)
    four_flat_plots('step_detect', m_accel_vector, 'acc', hf_acc_vec, 'hl', lf_acc_vec, 'lf', m_steps, 'steps')
    m_dy, m_rot_p, m_rot_y = vel_and_pos(m_steps, m_y, m_p, m_r, m_num)
    x_arr, y_arr, z_arr = points_for_track(m_dy, m_rot_p, m_rot_y, m_num)
    xy_view_plot(x_arr, y_arr)
    vol_plot(x_arr, y_arr, z_arr)


# read data from file
def read_data(filename):
    array = []
    accel_x = []
    accel_y = []
    accel_z = []
    accel_vector =[]
    gyro_x = []
    gyro_y = []
    gyro_z = []
    magn_x = []
    magn_y = []
    magn_z = []
    yaw = []
    pitch = []
    roll = []
    trueheading = []

    with open(filename, 'r') as file:
        data = file.readlines()
        for line in data:
            array.append([float(x) for x in line.split()])
        num = sum(1 for line in data)
        # print (num)
        for i in range(0, num):
            accel_vector.append(np.sqrt(array[i][1] ** 2 + array[i][2] ** 2 + array[i][3] ** 2))  # vector length of accel
            accel_x.append(array[i][1])
            accel_y.append(array[i][2])
            accel_z.append(array[i][3])
            #gyro_x.append(array[i][3])
            #gyro_y.append(array[i][4])
            #gyro_z.append(array[i][5])
            #magn_x.append(array[i][6])
            #magn_y.append(array[i][7])
            #magn_z.append(array[i][8])
            yaw.append(array[i][8] * pi / 180)  # from degrees to radians
            pitch.append(array[i][9] * pi / 180)
            roll.append(array[i][10] * pi / 180)
            if yaw[i] > 1: yaw[i] = 1
            if yaw[i] < -1: yaw[i] = -1
            if pitch[i] > 1: pitch[i] = 1
            if pitch[i] < -1: pitch[i] = -1
            if roll[i] > 1: roll[i] = 1
            if roll[i] < -1: roll[i] = -1
            #trueheading.append(atan2(magn_y[i], magn_x[i]))
    return accel_vector, accel_x, accel_y, accel_z, yaw, pitch, roll, num


# band-pass filter (a merge of low pass and high pass filters
def lp_and_hp_filter(array, num):
    # koeff for filters
    fc = 0.01  # 0.01
    b = 0.08  # 0.08
    N = int(np.ceil(0.5 / b))
    if not N % 2: N += 1
    n = np.arange(N)
    sample_period = 1 / 256

    # high pass filter
    sinc_func_h = np.sinc(2 * fc * (n - (N - 1) / 2.))
    window = np.blackman(N)
    sinc_func_h = sinc_func_h * window
    sinc_func_h = sinc_func_h / np.sum(sinc_func_h)
    high_filt_arr = np.convolve(array, sinc_func_h)

    arr_abs = np.fabs(high_filt_arr)  # the absolute value of an accel

    # low pass filter
    sinc_func_l = np.sinc(2 * fc * (n - (N - 1) / 2.))
    window = 0.42 - 0.5 * np.cos(2 * np.pi * n / (N - 1)) + 0.08 * np.cos(4 * np.pi * n / (N - 1))
    sinc_func_l = sinc_func_l * window
    sinc_func_l = sinc_func_l / np.sum(sinc_func_l)
    low_filt_arr = np.convolve(high_filt_arr, sinc_func_l)

    return high_filt_arr, low_filt_arr


# steps detection
def detect_step(lf_array, num):
    static = lf_array > 25
    static_ = static * 20
    '''for i in range(0, num):
        if array[i] < 1.5:
            static_array.append(array[i])'''
    return static_


# calculate velocity and positions
def vel_and_pos(array, y, p, r, num):
    velocity = []
    dist = []
    v = 0
    d = 0
    # calculate velocity from vector length of accel
    for i in range(0, num):
        v = v + array[i] * d_time
        velocity.append(v)
    velocity = np.array(velocity)
    dy = velocity * d_time

    for i in range(0, num):
        d = d + dy[i]
        dist.append(d)
    rot_angle_yaw = 2 * np.arcsin(y) * np.arcsin(p) * np.arcsin(r)
    rot_angle_pitch = 2 * np.arcsin(p)
    #rot_angle_roll = 2 * np.arcsin(r)
    return dy, rot_angle_pitch, rot_angle_yaw


# calculate points for trajectory
def points_for_track(dy, rot_angle_pitch, rot_angle_yaw,  num):
    x_arr = []
    y_arr = []
    z_arr = []
    x = 0
    y = 0
    z = 0
    for i in range(0, num - 1):
        x = x + dy[i + 1] * np.sin(rot_angle_yaw[i])
        y = y + dy[i + 1] * np.cos(rot_angle_yaw[i])
        z = z + dy[i + 1]  # * np.sin(rot_angle_pitch[i])*np.cos(rot_angle_yaw[i])
        x_arr.append(x)
        y_arr.append(y)
        z_arr.append(z)
    return x_arr, y_arr, z_arr

def one_flat_plot(plot_title, array0, name0):
    fig, plotting = plt.subplots()
    plotting.set_title(plot_title)
    line1, = plotting.plot(array0, color='red', label=name0)
    plt.xlabel("time step")
    plt.ylabel(plot_title)
    leg = plotting.legend(loc='upper left', fancybox=True, shadow=True)
    plt.grid()
    leg.get_frame().set_alpha(0.4)
    plt.show()


def three_flat_plots(plot_title, array0, name0, array1, name1, array2, name2):
    fig, plotting = plt.subplots()
    plotting.set_title(plot_title)
    line1, = plotting.plot(array0, label=name0)
    line2, = plotting.plot(array1, label=name1)
    line3, = plotting.plot(array2, label=name2)
    plt.xlabel("time step")
    plt.ylabel(plot_title)
    leg = plotting.legend(loc='upper left', fancybox=True, shadow=True)
    plt.grid()
    leg.get_frame().set_alpha(0.4)
    plt.show()


def four_flat_plots(plot_title, array0, name0, array1, name1, array2, name2, array3, name3):
    fig, plotting = plt.subplots()
    plotting.set_title(plot_title)
    line1, = plotting.plot(array0, label=name0)
    line2, = plotting.plot(array1, label=name1)
    line3, = plotting.plot(array2, label=name2)
    line4, = plotting.plot(array3, label=name3)
    plt.xlabel("time step")
    plt.ylabel(plot_title)
    leg = plotting.legend(loc='upper left', fancybox=True, shadow=True)
    plt.grid()
    leg.get_frame().set_alpha(0.4)
    plt.show()


def xy_view_plot(x, y):
    fig1, plotting = plt.subplots()
    plotting.set_title('Trajectory')
    plt.scatter(x, y, color='black')
    plt.xlabel("x-axis")
    plt.ylabel("y-axis")
    plt.show()


def vol_plot(x, y, z):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot(x, y, z, color='blue')
    ax.set_xlabel('X Label')
    ax.set_ylabel('Y Label')
    ax.set_zlabel('Z Label')
    plt.show()


if __name__ == '__main__':
    main()
