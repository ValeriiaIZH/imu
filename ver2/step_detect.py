'''
Python code which can represent the magnitude of person's step
There aren't functions in this code
Only lists were used
Also low_pass_filter and high_pass_filter were used
File dt 0_01.txt consist of values from accelerometre, gyroscope, magnetometre
and angles in such form:
№   accel_x  accel_y   accel_z  gyro_x  gyro_y  gyro_z  azimuth  yaw   pitch  roll
'''

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from math import *

# koeff for filters
fc = 0.01  #0.01
b = 0.08   #0.08
N = int(np.ceil(0.5 / b))
if not N % 2: N += 1
n = np.arange(N)
sample_period = 1/256

filename = "dt 0_01.txt"
d_time = 0.01
array = []
static_array = []
time = []
accel = []
gyro = []
magn = []

# read data from file
with open(filename, 'r') as file:
    data = file.readlines()
    for line in data:
        array.append([float(x) for x in line.split()])
    num = sum(1 for line in data)
    # print (num)
    for i in range(0, num):
        accel.append(np.sqrt(array[i][1]**2 + array[i][2]**2 + array[i][3]**2))
        gyro = [array[i][4], array[i][5], array[i][6]]
# high pass filter
sinc_func_h = np.sinc(2 * fc * (n - (N - 1) / 2.))
window = np.blackman(N)
sinc_func_h = sinc_func_h * window
sinc_func_h = sinc_func_h / np.sum(sinc_func_h)
high_filt_acc = np.convolve(accel, sinc_func_h)

acc_abs = np.fabs(high_filt_acc) # the absolute value of an accel

# low pass filter
sinc_func_l = np.sinc(2 * fc * (n - (N - 1) / 2.))
window = 0.42 - 0.5 * np.cos(2 * np.pi * n / (N - 1)) + 0.08 * np.cos(4 * np.pi * n / (N - 1))
sinc_func_l = sinc_func_l * window
sinc_func_l = sinc_func_l / np.sum(sinc_func_l)
low_filt_acc = np.convolve(high_filt_acc, sinc_func_l)

static = low_filt_acc > 25
static_ = static*20

fig, plotting = plt.subplots()
plotting.set_title('Acceleration (m/s^2)')
line1, = plotting.plot(accel,  color='red', label='acc')
line2, = plotting.plot(static_,  color='blue', label='stat')
line3, = plotting.plot(low_filt_acc, label='lf')
line4, = plotting.plot(high_filt_acc, label = 'hf')
plt.xlabel("time step")
plt.ylabel("Accelaration")
leg = plotting.legend(loc='upper left', fancybox=True, shadow=True)
plt.grid()
leg.get_frame().set_alpha(0.4)
plt.show()
