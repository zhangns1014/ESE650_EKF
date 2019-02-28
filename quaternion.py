import numpy as np
import math


def mul(a, b):
    t0 = a[0] * b[0] - a[1] * b[1] - a[2] * b[2] - a[3] * b[3]
    t1 = a[0] * b[1] + a[1] * b[0] + a[2] * b[3] - a[3] * b[2]
    t2 = a[0] * b[2] + a[2] * b[0] + a[3] * b[1] - a[1] * b[3]
    t3 = a[0] * b[3] + a[3] * b[0] - a[2] * b[1] + a[1] * b[2]
    return np.array([t0, t1, t2, t3])


def mulmat(a, X):
    t0 = a[0] * X[0, :] - a[1] * X[1, :] - a[2] * X[2, :] - a[3] * X[3, :]
    t1 = a[0] * X[1, :] + a[1] * X[0, :] + a[2] * X[3, :] - a[3] * X[2, :]
    t2 = a[0] * X[2, :] + a[2] * X[0, :] + a[3] * X[1, :] - a[1] * X[3, :]
    t3 = a[0] * X[3, :] + a[3] * X[0, :] - a[2] * X[1, :] + a[1] * X[2, :]
    return np.array([t0, t1, t2, t3])


def mulmat2(X, a):
    t0 = X[0, :] * a[0] - X[1, :] * a[1] - X[2, :] * a[2] - X[3, :] * a[3]
    t1 = X[0, :] * a[1] + X[1, :] * a[0] + X[2, :] * a[3] - X[3, :] * a[2]
    t2 = X[0, :] * a[2] - X[1, :] * a[3] + X[2, :] * a[0] + X[3, :] * a[1]
    t3 = X[0, :] * a[3] + X[1, :] * a[2] - X[2, :] * a[1] + X[3, :] * a[0]
    return np.array([t0, t1, t2, t3])


def inverse(a):
    n = float(np.sqrt(a[0] ** 2 + a[1] ** 2 + a[2] ** 2 + a[3] ** 2))
    return np.array([a[0], -a[1], -a[2], -a[3]]) / n


def quat(theta, omega):
    norm_omega = omega / float(np.sqrt(omega[0] ** 2 + omega[1] ** 2 + omega[2] ** 2))
    t = np.array(
        [np.cos(theta / 2.0), norm_omega[0] * np.sin(theta / 2.0), norm_omega[1] * np.sin(theta / 2.0),
         norm_omega[2] * np.sin(theta / 2.0)])
    return t


def new_quat(alpha, omega):
    q = np.zeros(4)
    q[0] = np.cos(alpha / 2.0)
    q[1] = np.sin(alpha / 2.0) * omega[0]
    q[2] = np.sin(alpha / 2.0) * omega[1]
    q[3] = np.sin(alpha / 2.0) * omega[2]
    return q


'''
quaternion to rotation vector 4 > 3
'''


def quat2rotationvector(q):
    # q=q/np.sqrt(np.sum(q**2))
    # if(q[0]==1):
    #     return np.array([0,0,0])
    # alpha = 2 * math.acos(q[0])
    # print(alpha)
    # return q[1:]/math.sin(alpha/2)*alpha
    # sin_alpha=np.linalg.norm(q[1:])
    # alpha=math.atan2(sin_alpha,q[0])*2
    # if sin_alpha == 0:
    #     return np.array([0,0,0])
    # return q[1:]/sin_alpha * alpha
    vec = q[1:].copy()
    return vec


'''
rotation vector to quaternion 3 > 4 
'''


def rotationvector2quat(X):
    Y = X.copy()
    alpha = float(np.sqrt(np.sum(X ** 2)))
    if alpha == 0:
        return np.array([1, 0, 0, 0])
    Y = X / float(alpha) * np.sin(alpha / 2.0)
    return np.array([np.cos(alpha / 2.0), Y[0], Y[1], Y[2]])


def quat2rpy(q):
    t0 = math.atan2(2 * (q[0] * q[1] + q[2] * q[3]), 1 - 2 * (q[1] ** 2 + q[2] ** 2))
    t1 = math.asin(2 * (q[0] * q[2] - q[3] * q[1]))
    t2 = math.atan2(2 * (q[0] * q[3] + q[1] * q[2]), 1 - 2 * (q[2] ** 2 + q[3] ** 2))
    return np.array([t0, t1, t2])
