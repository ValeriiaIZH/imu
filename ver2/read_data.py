from math import *
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from mpl_toolkits.mplot3d import Axes3D
from numpy import dot, sum, tile, linalg
from numpy.linalg import inv, pinv
import pandas as pd

d_time = 0.001


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
    m_accel_vector, m_ax, m_ay, m_az, m_gx, m_gy, m_gz, m_y, m_p, m_r, m_num = read_data(filename)
    hf_acc_vec, lf_acc_vec = lp_and_hp_filter(m_accel_vector, m_num)
    m_steps = detect_step(lf_acc_vec, m_num)
    four_flat_plots('step_detect', m_accel_vector, 'acc', hf_acc_vec, 'hl', lf_acc_vec, 'lf', m_steps, 'steps')

    m_dy, m_velocity, m_rot_p, m_rot_y = vel_and_pos(lf_acc_vec, m_y, m_p, m_r, m_num)
    one_flat_plot('velocity', m_velocity, 'vel')
    x_arr, y_arr, z_arr = points_for_track(m_dy, m_steps, m_rot_p, m_rot_y, m_num)
    xy_view_plot(x_arr, y_arr)
    vol_plot(x_arr, y_arr, z_arr)
    m_rotat, path_x_kalman, path_y_kalman, path_z_kalman, m_posx, m_posy, m_posz, steps_kalman = kalman_fiter(m_ax, m_ay, m_az, m_gx, m_gy, m_gz, m_num)
    x_kalman, y_kalman = kalman_coordinares(steps_kalman, m_rotat)
    vol_plot(x_kalman, y_kalman, path_z_kalman)


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
            gyro_x.append(array[i][4])
            gyro_y.append(array[i][5])
            gyro_z.append(array[i][6])
            # magn_x.append(array[i][6])
            # magn_y.append(array[i][7])
            # magn_z.append(array[i][8])
            yaw.append(array[i][8] * pi / 180)  # from degrees to radians
            pitch.append(array[i][9] * pi / 180)
            roll.append(array[i][10] * pi / 180)
            if yaw[i] > 1: yaw[i] = 1
            if yaw[i] < -1: yaw[i] = -1
            if pitch[i] > 1: pitch[i] = 1
            if pitch[i] < -1: pitch[i] = -1
            if roll[i] > 1: roll[i] = 1
            if roll[i] < -1: roll[i] = -1
            # trueheading.append(atan2(magn_y[i], magn_x[i]))
    return accel_vector, accel_x, accel_y, accel_z, gyro_x, gyro_y, gyro_z, yaw, pitch, roll, num


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
    # rot_angle_roll = 2 * np.arcsin(r)
    """when array < 2.5 velocity = 0"""
    return dy, velocity, rot_angle_pitch, rot_angle_yaw


# calculate points for trajectory
def points_for_track(dy, static, rot_angle_pitch, rot_angle_yaw, num):
    x_arr = []
    y_arr = []
    z_arr = []
    x = 0
    y = 0
    z = 0
    for i in range(0, num):
        if static[i] < 10:
            dy[i] = 0

    for i in range(0, num - 1):
        x = x + dy[i + 1] * np.sin(rot_angle_yaw[i])
        y = y + dy[i + 1] * np.cos(rot_angle_yaw[i])
        z = z + dy[i + 1]
        x_arr.append(x)
        y_arr.append(y)
        z_arr.append(z)

    return x_arr, y_arr, z_arr


# kalman filter for accelerometre
def kalman_fiter(acc_x, acc_y, acc_z, gyr_x, gyr_y, gyr_z, num):
    distance = 0
    movement_x = 0
    movement_y = 0
    movement_z = 0

    path_x = []
    path_y = []
    path_z = []
    steps_path = []

    for i in range(num):
        if i == 0:
            X = np.zeros((3, 1))
            P = np.eye(3)
            A = np.eye(3)
            Q = np.array([1e-5, 1e-5, 1e-5])
            R = np.array([5e-1, 5e-1, 1e-2])
            B = np.zeros((3, 3))
            U = np.zeros((3, 1))
            H = np.eye(3)

            acc_kalman = kalman(np.array(X), np.array(P), np.array(A), np.array(Q), np.array(R), np.array(B), np.array(U), np.array(H))

            distance_x = acc_x[i] * (d_time ** 2) * 0.5
            distance_y = acc_y[i] * (d_time ** 2) * 0.5
            distance_z = acc_z[i] * (d_time ** 2) * 0.5

        else:

            acc_kalman.predict(np.zeros((3, 1)))
            old_acc = np.array([[acc_x[i]], [acc_y[i]], [acc_z[i]]])
            new_acc = acc_kalman.update(old_acc)

            distance_x = new_acc[0][0] * (d_time ** 2) * 0.5
            distance_y = new_acc[1][0] * (d_time ** 2) * 0.5
            distance_z = old_acc[2][0] * (d_time ** 2) * 0.5

        distance += np.sqrt(distance_x ** 2 + distance_y ** 2)
        steps_path.append(np.sqrt(distance_x ** 2 + distance_y ** 2))

        movement_x += distance_x
        movement_y += distance_y
        movement_z += -1 * distance_z

        path_x.append(movement_x)
        path_y.append(movement_y)
        path_z.append(movement_z)

        rX = gyr_x
        rY = gyr_y
        rZ = gyr_z

        angleX = 0
        angleY = 0
        angleZ = 0

        poseX = []
        poseY = []
        poseZ = []

        rotations = []

        for i in range(num):
            angleX += rX[i] * d_time
            angleY += rY[i] * d_time

            if i == 0:
                X = 0
                P = 1
                A = 1
                Q = 1e-5
                R = 3e-1
                B = 0
                U = 0
                H = 1

                angleZKalman = kalman(np.array(X), \
                                      np.array(P), \
                                      np.array(A), \
                                      np.array(Q), \
                                      np.array(R), \
                                      np.array(B), \
                                      np.array(U), \
                                      np.array(H))
                angleZ += rZ[i] * d_time
                rotations.append(angleZ)

            else:
                angleZKalman.predict(np.array(0))
                rotationRate = angleZKalman.update(np.array(rZ[i]))
                rotation = rotationRate * d_time
                angleZ += rotation
                rotations.append(rotation)

            poseX.append(angleX)
            poseY.append(angleY)
            poseZ.append(angleZ)
    angleX = 0
    angleY = 0
    angleZ = 0

    poseX = []
    poseY = []
    poseZ = []

    rotations = []

    for i in range(num):
        angleX += rX[i] * d_time
        angleY += rY[i] * d_time

        if i == 0:
            X = 0
            P = 1
            A = 1
            Q = 1e-5
            R = 3e-1
            B = 0
            U = 0
            H = 1

            angleZKalman = kalman(np.array(X), \
                                  np.array(P), \
                                  np.array(A), \
                                  np.array(Q), \
                                  np.array(R), \
                                  np.array(B), \
                                  np.array(U), \
                                  np.array(H))
            angleZ += rZ[i] * d_time
            rotations.append(angleZ)

        else:
            angleZKalman.predict(np.array(0))
            rotationRate = angleZKalman.update(np.array(rZ[i]))
            rotation = rotationRate * d_time
            angleZ += rotation
            rotations.append(rotation)

        poseX.append(angleX)
        poseY.append(angleY)
        poseZ.append(angleZ)
    return rotations, path_x, path_y, path_z, poseX, poseY, poseZ, steps_path


# coordinates
def kalman_coordinares(steps_path, rotations):
    x = [0, steps_path[1]]
    y = [0, 0]

    for i in range(1, len(rotations) - 1):
        angle = rotations[i + 1]
        radius = steps_path[i + 1]

        slope = (y[i] - y[i - 1]) / (x[i] - x[i - 1])

        x1 = x[i] + radius * np.sqrt(1 / (1 + slope ** 2))
        x2 = x[i] - radius * np.sqrt(1 / (1 + slope ** 2))

        y1 = y[i] + slope * radius * np.sqrt(1 / (1 + slope ** 2))
        y2 = y[i] - slope * radius * np.sqrt(1 / (1 + slope ** 2))

        valid = [0, 0, 0, 0]
        if abs((y1 - y[i]) / (x1 - x[i]) - slope) < 1e-6:
            valid[0] = 1
        if abs((y1 - y[i]) / (x2 - x[i]) - slope) < 1e-6:
            valid[1] = 1
        if abs((y2 - y[i]) / (x1 - x[i]) - slope) < 1e-6:
            valid[2] = 1
        if abs((y2 - y[i]) / (x2 - x[i]) - slope) < 1e-6:
            valid[3] = 1

        for j, v in enumerate(valid):
            if v == 1:
                if j == 0 or j == 2:
                    if (x[i - 1] < x[i] and x1 < x[i]) or (x[i - 1] > x[i] and x1 > x[i]):
                        valid[0] = 0
                        valid[2] = 0
                if j == 1 or j == 3:
                    if (x[i - 1] < x[i] and x2 < x[i]) or (x[i - 1] > x[i] and x2 > x[i]):
                        valid[1] = 0
                        valid[3] = 0

        end_x = 0
        end_y = 0

        for j, v in enumerate(valid):
            if v == 1:
                if j == 0 or j == 2:
                    end_x = x1
                    end_y = y1
                    break
                if j == 1 or j == 3:
                    end_x = x2
                    end_y = y2
                    break

        rotated_coords = dot(np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]]), \
                            np.array([[end_x - x[i]], [end_y - y[i]]])) + np.array([[x[i]], [y[i]]])

        x.append(rotated_coords[0][0])
        y.append(rotated_coords[1][0])
    return x, y


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
