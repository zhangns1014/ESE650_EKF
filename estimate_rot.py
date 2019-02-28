# data files are numbered on the server.
# for exmaple imuRaw1.mat, imuRaw2.mat and so on.
# write a function that takes in an input number (1 through 6)
# reads in the corresponding imu Data, and estimates
# roll pitch and yaw using an extended kalman filter

import scipy.io as sio
import numpy as np
import matplotlib.pyplot as plt
from helper import rotationtoangle, update
from quaternion import quat2rpy


def estimate_rot(data_num):
    # print("data_num " , data_num)
    # your code goes here
    # load data
    s = "./imu/imuRaw" + str(data_num)
    imu = sio.loadmat(s)
    data_imu = imu['vals']

    # bias and scale
    ax_bias = np.sum(data_imu[0, :800]) / 800
    ay_bias = np.sum(data_imu[1, :800]) / 800
    az_bias = (ax_bias + ay_bias) / 2
    a_data = np.sum((data_imu[0:3, :] - np.array([[ax_bias], [ay_bias], [az_bias]])) ** 2, axis=0)
    a_scale = np.sum(np.sqrt(a_data)) / np.sum(a_data)
    print(a_scale)
    s_g = 3.3
    g_scale = 3300 / 1023 / s_g / 180 * np.pi
    print(g_scale)
    wx_bias = np.mean(data_imu[4, :100])
    wy_bias = np.mean(data_imu[5, :100])
    wz_bias = np.mean(data_imu[3, :100])
    # process data
    a_meassure = (data_imu[0:3, :] - np.array([[ax_bias], [ay_bias], [az_bias]])) * a_scale
    a_meassure[0, :] = -a_meassure[0, :]
    a_meassure[1, :] = -a_meassure[1, :]
    w_meassure = (np.array([data_imu[4, :], data_imu[5, :], data_imu[3, :]]) - np.array(
        [[wx_bias], [wy_bias], [wz_bias]])) * g_scale
    # print(a_meassure)
    # print(w_meassure)
    delta_t = imu['ts'][0, 1:] - imu['ts'][0, 0:-1]
    delta_t=np.hstack((0,delta_t))
    P_k_1 = np.eye(6) * 10
    Q = np.eye(6)  # process noise (6x6)
    Q[:3, :3] += np.ones(3)
    Q[3:, 3:] += np.ones(3)
    Q[:3, :3] *= 5e-8
    R = np.eye(6)  # measurement noise (6x6)
    R[:3, :3] += np.ones(3)
    R[3:, 3:] += np.ones(3)
    R[:3, :3] *= 10e-4
    R[3:, 3:] *= 10e-6
    x_k_1 = np.array([1, 0, 0, 0, 0, 0, 0])
    z = np.vstack((a_meassure, w_meassure))
    T = delta_t.shape[0]
    r = np.zeros(T)
    p = np.zeros(T)
    y = np.zeros(T)
    for i in range(T):
        # print(i,"started")
        x_k, P_k = update(P_k_1, Q, R, x_k_1, delta_t[i], z[:, i])
        x_k_1 = x_k
        P_k_1 = P_k
        r[i], p[i], y[i] = quat2rpy(x_k)
    # print("it's here !")
    return r, p, y


if __name__ == "__main__":
    rot1 = sio.loadmat('imu/viconRot2.mat')
    imu1 = sio.loadmat('imu/imuRaw2.mat')
    data_rots = rot1['rots']
    angle_vicon = rotationtoangle(data_rots)
    t_rots = rot1['ts'].reshape(-1)

    r,p,y=estimate_rot(2)
    ax = plt.subplot(3, 1, 1)
    ax.plot(t_rots, angle_vicon[:, 0])
    ax.plot(imu1['ts'][0,:], r)
    ax = plt.subplot(3, 1, 2)
    ax.plot(t_rots, angle_vicon[:, 1])
    ax.plot(imu1['ts'][0,:], p)
    ax = plt.subplot(3, 1, 3)
    ax.plot(t_rots, angle_vicon[:, 2])
    ax.plot(imu1['ts'][0,:], y)

    plt.show()
