import numpy as np
import math
from quaternion import mulmat, quat2rotationvector, mulmat2, quat2rpy, inverse, mul,rotationvector2quat, new_quat
from scipy.linalg import cholesky, sqrtm


def isRotationMatrix(R) :
    Rt = np.transpose(R)
    shouldBeIdentity = np.dot(Rt, R)
    I = np.identity(3, dtype = R.dtype)
    n = np.linalg.norm(I - shouldBeIdentity)
    return n < 1e-6


# Calculates rotation matrix to euler angles
# The result is the same as MATLAB except the order
# of the euler angles ( x and z are swapped ).
def rotationMatrixToEulerAngles(R) :

    # assert isRotationMatrix(R)

    sy = math.sqrt(R[0 ,0] * R[0 ,0] +  R[1 ,0] * R[1 ,0])

    singular = sy < 1e-6

    if not singular :
        x = math.atan2(R[2 ,1] , R[2 ,2])
        y = math.atan2(-R[2 ,0], sy)
        z = math.atan2(R[1 ,0], R[0 ,0])
    else :
        x = math.atan2(-R[1 ,2], R[1 ,1])
        y = math.atan2(-R[2 ,0], sy)
        z = 0

    return np.array([x, y, z])


def rotationtoangle(R_N):
    res=np.array([[0,0,0]])
    for i in range(R_N.shape[2]):
        temp=rotationMatrixToEulerAngles(R_N[:,:,i])
        res=np.vstack((res,temp))
    res=res[1:,:]
    return res


def isdefinite(mat):
    return np.all(np.linalg.eigvals(mat) > 0)


def decompose(mat):
    # print("decompose!")
    # assert isdefinite(mat)
    # print(mat)
    # print("yes!")
    return cholesky(mat)


def norm(vec):
    return np.sqrt(np.sum(vec**2))


def generate_sigma(P_k_1, Q, x_k_1):
    # print("P_k_1 in generate sigma! ", P_k_1)
    S1=decompose(P_k_1 + Q)*np.sqrt(12)
    S2=-S1
    S1=np.hstack((S1,S2))
    q_w=np.zeros([4,12])
    for i in range(12):
        q_w[:,i]=rotationvector2quat(S1[0:3,i])
    X_q=mulmat(x_k_1[0:4], q_w)
    X_w = x_k_1[4:].reshape(-1,1) + S1[3:, :]
    return X_q,X_w


def state_transfer(X_q, X_w, x_w_1,delta_t):
    """
    :param X_q: quaternion
    :param X_w:  angular velocity
    :param x_w_1: prev angular velocity
    :param delta_t:
    :return:
    """
    X_1q=np.zeros([4,12])
    if norm(x_w_1) == 0:
        q_delta = np.array([1, 0, 0, 0])
    else:
        alpha_delta = norm(x_w_1) * delta_t
        e_delta = x_w_1 / float(norm(x_w_1))
        q_delta = new_quat(alpha_delta, e_delta)
    for i in range(12):
        X_1q[:,i] = mul(X_q[:,i], q_delta)  # quaternion points after transformation
    return X_1q, X_w


def get_average(qt_begin, Qt, thres_e = 0.001, thres_num=50):
    err=1000
    num=0
    qt=qt_begin
    while err >= thres_e and num < thres_num:
        qt_inv = inverse(qt)
        e = np.zeros(3)
        for i in range(12):
            e_i=mul(Qt[:,i],qt_inv)
            e+=quat2rotationvector(e_i)
        e=e/12.0
        err=norm(e)
        num+=1
        e_j=rotationvector2quat(e)
        qt=mul(e_j,qt)
    return qt


def get_rw(Qt, q_avg, X_w):
    rw = mulmat2(Qt,inverse(q_avg))
    W=np.zeros([3,12])
    for i in range(12):
        W[:,i]=quat2rotationvector(rw[:,i])
    W = np.vstack((W, X_w-np.sum(X_w,axis=1).reshape(-1,1)/12.0))
    return W


def get_cov_pk_(W):  #omega_avg is a 1d array
    return W.dot(W.T)/12.0

'''
now calculate Z_i Hi
Z_q 3*12 measurement
W 3*12 
'''
def Z_transfer(Qt,X_w):
    Z_q=np.zeros([3,12])
    for i in range(12):
        Z_q[:,i]=quat2rotationvector(mul(Qt[:,i],mul(np.array([0,0,0,1]),inverse(Qt[:,i]))))
    return np.vstack((Z_q,X_w))


def z_mean_cov(Z):
    z_avg=np.sum(Z,axis=1)/12.0
    Z_diff=Z-z_avg.reshape(-1,1)
    P_zz=1/12.0*(Z_diff).dot((Z_diff).T)
    return z_avg, P_zz


def get_P_xz(W,Z,z_avg):
    return W.dot((Z-z_avg.reshape(-1,1)).T)/12.0


def get_kalman_gain(P_xz, P_zz, R):
    return P_xz.dot(np.linalg.inv(P_zz+R))

def get_x_k(xk_pred,K,z_k,z_avg):
    err=np.dot(K,(z_k-z_avg).reshape(-1,1)).reshape(-1)
    err_quat=rotationvector2quat(err[0:3])
    xk=np.zeros(7)
    xk[0:4]=mul(xk_pred[0:4],err_quat)
    xk[4:]=xk_pred[4:]+err[3:]
    return xk


def update(P_k_1, Q, R, x_k_1, delta_t, z_k):
    # print("P_k_1" ,P_k_1)
    # print(x_k_1)
    X_q, X_w = generate_sigma(P_k_1, Q, x_k_1)
    # print("after generate sigma points")
    Y_q, Y_w = state_transfer(X_q, X_w, x_k_1[4:],delta_t)
    # print(Y_q, Y_w)
    Y_w_avg=np.sum(Y_w,axis=1)/12.0
    # print("average yw ",Y_w_avg)
    Y_q_avg=get_average(x_k_1[0:4],Y_q)
    # print("average yq ",Y_q_avg)
    W=get_rw(Y_q, Y_q_avg, Y_w)
    # print("get rw ", W)
    P_k_1=get_cov_pk_(W)
    # print("get pk-1", P_k_1)
    xk_pred = np.hstack((Y_q_avg,Y_w_avg))
    # print("xk_pred ",xk_pred)
    Z=Z_transfer(Y_q,Y_w)
    # print("z_transfer ", Z)
    z_avg, P_zz=z_mean_cov(Z)
    # print("z_avg ",z_avg)
    # print("P_zz", P_zz)
    P_xz = get_P_xz(W,Z,z_avg)
    # print("p_xz ",P_xz)
    K = get_kalman_gain(P_xz, P_zz, R)
    # print("get kalman gain")
    P_k=P_k_1-K.dot(P_zz+R).dot(K.T)
    x_k= get_x_k(xk_pred,K,z_k,z_avg)
    # print("update xk")
    return x_k,P_k



