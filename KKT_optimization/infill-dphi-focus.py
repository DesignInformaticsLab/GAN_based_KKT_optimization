import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import scipy.io as sio
from tensorflow.contrib import rnn
import math
import time
import os
from datetime import datetime
import dateutil.tz
import time
import copy

# from utils.utils import *


def xavier_init(size):
    in_dim = size[0]
    xavier_stddev = 1. / tf.sqrt(in_dim / 2.)
    return tf.random_normal(shape=size, stddev=xavier_stddev)

def creat_dir(network_type):
    """code from on InfoGAN"""
    now = datetime.now(dateutil.tz.tzlocal())
    timestamp = now.strftime('%Y_%m_%d_%H_%M_%S')
    root_log_dir = "logs/" + network_type
    exp_name = network_type + "_%s" % timestamp
    log_dir = os.path.join(root_log_dir, exp_name)

    now = datetime.now(dateutil.tz.tzlocal())
    timestamp = now.strftime('%Y_%m_%d_%H_%M_%S')
    root_model_dir = "models/" + network_type
    exp_name = network_type + "_%s" % timestamp
    model_dir = os.path.join(root_model_dir, exp_name)

    for path in [log_dir, model_dir]:
        if not os.path.exists(path):
            os.makedirs(path)
    return log_dir, model_dir

tic = time.clock()

E0 = 1
Emin = 1e-9
nu = 0.3
batch_size = 1
directory_data='experiment_data'

'Input'
nelx, nely, alpha, alpha2, gamma, rmin, density_r = 12*4, 4*4, 0.6, 0.6, 3.0, 3.0, 6.0

'Algorithm parameters'
p, nn, epsilon_al, epsilon_opt, beta = 16,nely * nelx, 1, 1e-3, 8

'Prepare filter'
r = rmin
Range = np.arange(-r, r + 1)
X = np.array([Range] * len(Range))
Y = np.array([Range] * len(Range)).T
X_temp = X.T.reshape(len(Range) ** 2, 1)
Y_temp = Y.T.reshape(len(Range) ** 2, 1)
neighbor = np.concatenate((X_temp, Y_temp), 1)
D = np.sum(neighbor ** 2, 1)
rn = sum(D <= r ** 2) - sum(D == 0)

pattern_index = []
for i in range(len(D)):
    if D[i] <= r ** 2 and D[i] != 0:
        pattern_index.append(i)
pattern = neighbor[pattern_index]

locX = np.asarray([np.arange(1, nelx + 1)] * nely)
locY = np.asarray([np.arange(1, nely + 1)] * nelx).T
loc = np.asarray([locY.T.reshape(-1), locX.T.reshape(-1)]).T

M = np.zeros([nn, rn])
for i in range(nn):
    for j in range(rn):
        locc = loc[i, :] + pattern[j, :]
        count_true = 0
        if locc[0] > nely: count_true = count_true + 1
        if locc[1] > nelx: count_true = count_true + 1

        if sum(locc < 1) + count_true == 0:
            M[i, j] = locc[0] + (locc[1] - 1) * nely
        else:
            M[i, j] = 'NaN'

iH = np.ones([int(nelx * nely * (2 * (np.ceil(rmin) - 1) + 1) ** 2), 1], dtype=int)
jH = np.ones(iH.shape, dtype=int)
sH = np.zeros(iH.shape)
k = 0
for i1 in range(nelx):
    for j1 in range(nely):
        e1 = (i1) * nely + (j1 + 1)
        for i2 in np.arange(int(max(i1 + 1 - (np.ceil(rmin) - 1), 1)),
                            int(min(i1 + 1 + (np.ceil(rmin) - 1), nelx)) + 1):
            for j2 in np.arange(int(max(j1 + 1 - (np.ceil(rmin) - 1), 1)),
                                int(min(j1 + 1 + (np.ceil(rmin) - 1), nely)) + 1):
                e2 = (i2 - 1) * nely + j2
                iH[k] = e1
                jH[k] = e2
                sH[k] = max(0, rmin - np.sqrt((i1 + 1 - i2) ** 2 + (j1 + 1 - j2) ** 2))
                k = k + 1

iH = iH.reshape(-1) - 1
jH = jH.reshape(-1) - 1
sH = sH.reshape(-1)

import scipy.sparse as sps

H = sps.csc_matrix((sH, (iH, jH))).toarray()
Hs = np.sum(H, 1)
bigM = H > 0

'create neighbourhood index for N'
r = density_r
Range = np.arange(-r, r + 1)
mesh = Range
X = np.asarray([Range] * len(Range))
Y = np.asarray([Range] * len(Range)).T
neighbor = np.array([X.T.reshape(-1), Y.T.reshape(-1)]).T
D = np.sum(neighbor ** 2, 1)
rn = np.sum(D <= r ** 2)
pattern = neighbor[D <= r ** 2, :]
locX = np.asarray([np.arange(1, nelx + 1)] * nely)
locY = np.asarray([np.arange(1, nely + 1)] * nelx).T
loc = np.asarray([locY.T.reshape(-1), locX.T.reshape(-1)]).T
N = np.zeros([nn, rn])
for i in range(nn):
    for j in range(rn):
        locc = loc[i, :] + pattern[j, :]
        count_true = 0
        if locc[0] > nely: count_true = count_true + 1
        if locc[1] > nelx: count_true = count_true + 1

        if sum(locc < 1) + count_true == 0:
            N[i, j] = locc[0] + (locc[1] - 1) * nely
        else:
            N[i, j] = 'NaN'

idx_temp = (np.array([np.arange(nn)] * rn).T).reshape(-1).reshape(nn * rn, 1)
N_t = N.T
idy = []
idx = []
import math

k = 0
for i in range(N_t.shape[1]):
    for j in range(N_t.shape[0]):
        if not math.isnan(N_t[j, i]):
            idy.append(N_t[j, i])
            idx.append(idx_temp[k])
        k = k + 1
idy = np.asarray(idy).reshape(len(idy), 1) - 1
idx = np.asarray(idx)
bigN = sps.coo_matrix((np.ones(len(idx)), (idx.reshape(-1), idy.reshape(-1)))).toarray()
N_count = np.sum(~np.isnan(N), axis=1)

'Material Properties'
E0, Emin, nu = 1, 1e-9, 0.3

'PREPARE FINITE ELEMENT ANALYSIS'
A11 = np.array([[12, 3, -6, -3], [3, 12, 3, 0], [-6, 3, 12, -3], [-3, 0, -3, 12]])
A12 = np.array([[-6, -3, 0, 3], [-3, -6, -3, -6], [0, -3, -6, 3], [3, -6, 3, -6]])
B11 = np.array([[-4, 3, -2, 9], [3, -4, -9, 4], [-2, -9, -4, -3], [9, 4, -3, -4]])
B12 = np.array([[2, -3, 4, -9], [-3, 2, 9, -2], [4, 9, 2, 3], [-9, -2, 3, 2]])

A1 = np.concatenate((A11, A12), axis=1)
A2 = np.concatenate((A12.T, A11), axis=1)
A = np.concatenate((A1, A2))

B1 = np.concatenate((B11, B12), axis=1)
B2 = np.concatenate((B12.T, B11), axis=1)
B = np.concatenate((B1, B2))

KE = 1 / (1 - nu ** 2) / 24. * (A + nu * B)
nodenrs = np.arange((1 + nelx) * (1 + nely)).reshape(1 + nelx, 1 + nely).T
edofVec = (2 * (nodenrs[0:-1, 0:-1] + 1)).reshape(nelx * nely, 1, order='F')
edofMat = np.asarray([edofVec.reshape(-1)] * 8).T + \
          np.asarray(([0, 1, 2 * nely + 2, 2 * nely + 3, 2 * nely + 0, 2 * nely + 1, -2, -1]) * nelx * nely).reshape(
              nelx * nely, 8)

iK = np.zeros([edofMat.shape[0] * 8, edofMat.shape[1] * 1])
for i in range(edofMat.shape[0]):
    for j in range(edofMat.shape[1]):
        iK[i * 8:(i + 1) * 8, j] = np.asarray(edofMat[i, j] * np.ones([8, 1])).reshape(-1)
iK = (iK.astype(np.int32).T).reshape(64 * nelx * nely, 1, order='F')

jK = np.zeros([edofMat.shape[0] * 1, edofMat.shape[1] * 8])
for i in range(edofMat.shape[0]):
    for j in range(edofMat.shape[1]):
        jK[i, j * 8:(j + 1) * 8] = np.asarray(edofMat[i, j] * np.ones([1, 8])).reshape(-1)
jK = (jK.astype(np.int32).T).reshape(64 * nelx * nely, 1, order='F')

'DEFINE LOADS AND SUPPORTS (HALF MBB-BEAM)'
U = np.zeros([2 * (nely + 1) * (nelx + 1), 1])

fixeddofs = np.asarray(np.arange(0, 2 * (nely + 1), 1).tolist())
fixeddofs = fixeddofs.reshape(1, len(fixeddofs))
alldofs = np.arange(2 * (nely + 1) * (nelx + 1)).reshape(1, 2 * (nely + 1) * (nelx + 1))

freedofs = np.setdiff1d(alldofs, fixeddofs)

'prepare some stuff to reduce cost'
dphi_idphi = (H / sum(H)).T

##############################################################################
'Network Structure'
z_dim, h_dim_1, h_dim_2, h_dim_3, h_dim_4, h_dim_5 = 3, 100, 100, 100, 100, 100

F_input = tf.placeholder(tf.float32, shape=([batch_size, z_dim]))
F = tf.placeholder(tf.float32, shape=([2*(nely+1)*(nelx+1),batch_size]))

W1 = tf.Variable(xavier_init([z_dim, h_dim_1]))
b1 = tf.Variable(tf.zeros(shape=[h_dim_1]))

W2 = tf.Variable(xavier_init([h_dim_1, h_dim_2]))
b2 = tf.Variable(tf.zeros(shape=[h_dim_2]))

W3 = tf.Variable(xavier_init([h_dim_2, h_dim_3]))
b3 = tf.Variable(tf.zeros(shape=[h_dim_3]))

W4 = tf.Variable(xavier_init([h_dim_3, h_dim_4]))
b4 = tf.Variable(tf.zeros(shape=[h_dim_4]))

W5 = tf.Variable(xavier_init([h_dim_4, nn]))
b5 = tf.Variable(tf.zeros(shape=[nn]))

h1 = tf.contrib.layers.batch_norm(tf.nn.relu(tf.matmul(F_input, W1) + b1),scale=True)
h2 = tf.contrib.layers.batch_norm(tf.nn.relu(tf.matmul(h1, W2) + b2),scale=True)
h3 = tf.contrib.layers.batch_norm(tf.nn.relu(tf.matmul(h2, W3) + b3),scale=True)
h4 = tf.contrib.layers.batch_norm(tf.nn.relu(tf.matmul(h3, W4) + b4),scale=True)
# h5 = tf.nn.relu(tf.matmul(h4, W5) + b5)
    
phi_ = tf.sigmoid(tf.matmul(h4, W5) + b5)
phi = tf.reshape(phi_, [nn, batch_size])
##################################################################################################
# phi = (tf.Variable(alpha * np.ones([nn, 1]), dtype='float32'))


sep_grad_store,error_store,dphi_store,dobj_store=[],[],[],[]
c,g,global_density=[],[],[]
rho=[]
dphi_fake=[]
error_store=[]


for i in range(batch_size):
    phi_til = tf.matmul(tf.cast(H, tf.float32), phi[:,i:i+1]) / Hs.reshape(nn, 1)
    rho.append((tf.tanh(beta / 2.0) + tf.tanh(beta * (phi_til - 0.5))) / (2 * tf.tanh(beta / 2.0)))

    sK_temp = tf.transpose((KE.reshape(KE.shape[0] * KE.shape[1], 1) * \
                            tf.reshape(Emin + tf.reshape(rho[i], [-1]) ** (gamma) * (E0 - Emin), [1, nelx * nely])))
    sK = tf.reshape(sK_temp, [8 * 8 * nelx * nely, 1])

    ###################
    indices_m = np.stack((iK.reshape(-1), jK.reshape(-1)), axis=1)
    values_m = tf.reshape(sK, [-1])

    linearized_m = tf.matmul(indices_m, [[10000], [1]])
    y_m, idx_m = tf.unique(tf.squeeze(linearized_m))

    idx_m_sort, ind_m_sort = tf.nn.top_k(idx_m, k=nelx * nely * 64)
    idx_m_sort = tf.reverse(idx_m_sort, [0])
    ind_m_sort = tf.reverse(ind_m_sort, [0])

    # values_m_test = values_m
    values_m = tf.gather(values_m, ind_m_sort)
    values_m = tf.segment_sum(values_m, idx_m_sort)

    y_m = tf.expand_dims(y_m, 1)
    indices_m = tf.concat([y_m // 10000, y_m % 10000], axis=1)
    #####################

    #####################
    K_sp = tf.SparseTensor(tf.cast(indices_m, tf.int64),
                           tf.reshape(tf.cast(values_m, tf.float32), [-1]),
                           [(nely + 1) * (nelx + 1) * 2, (nely + 1) * (nelx + 1) * 2])
    K_sp = tf.sparse_add(tf.zeros(((nely + 1) * (nelx + 1) * 2, (nely + 1) * (nelx + 1) * 2)), K_sp)
    K_dense = (K_sp + tf.transpose(K_sp)) / 2
    #####################

    K_temp = tf.gather(K_dense, freedofs.astype(np.int32))
    K_free = tf.transpose(tf.gather(tf.transpose(K_temp), freedofs.astype(np.int32)))
    
    F_RHS=tf.cast(tf.gather(tf.reshape(F[:,i:i+1], [(nelx + 1) * (nely + 1) * 2, 1]), freedofs.astype(np.int64)),
                              tf.float32)
    chol = tf.cholesky(K_free+np.diag((np.ones([1,(nelx+1)*(nely+1)*2-len(fixeddofs[0])])*1e-8).tolist()[0]))
    U_pre = tf.cholesky_solve(chol,F_RHS)
    
#     U_pre = tf.matmul(tf.matrix_inverse(tf.cast(K_free, tf.float32)),
#                       tf.cast(tf.gather(tf.reshape(F[:,i:i+1], [(nelx + 1) * (nely + 1) * 2, 1]), freedofs.astype(np.int64)),
#                               tf.float32))

    U = tf.sparse_add(tf.zeros(((nelx + 1) * (nely + 1) * 2)),
                      tf.SparseTensor(freedofs.reshape((nelx + 1) * (nely + 1) * 2 - (nely + 1) * 2, 1),
                      tf.reshape(tf.cast(U_pre, tf.float32), [-1]), [(nelx + 1) * (nely + 1) * 2]))

    'Objective Function and Sensitivity Analysis'
    ce = tf.reduce_sum(tf.multiply(tf.matmul(tf.reshape(tf.gather(U, edofMat.astype(np.int32)), edofMat.shape),
                                             KE.astype(np.float32)), tf.gather(U, edofMat.astype(np.int32))), axis=1)
    ce = tf.reshape(ce, [nn, 1])
    c.append(tf.reduce_sum(tf.multiply(tf.cast(tf.multiply(rho[i] ** gamma, (E0 - Emin)), tf.float32), ce)))
    rho_bar = tf.divide(tf.matmul(tf.cast(bigN, tf.float32), rho[i]), N_count.reshape(nn, 1))
    g.append((tf.reduce_sum(rho_bar ** p) / nely / nelx) ** (1.0 / p) / alpha - 1.0)
    global_density.append(tf.matmul(tf.transpose(rho[i]), tf.ones([nn, 1])) / nn - alpha2)
    ###################################################################
    dc_drho = -gamma * rho[i] ** (gamma - 1) * (E0 - Emin) * ce
    drho_dphi = beta * (1 - tf.tanh(beta * (phi[:,i:i+1] - 0.5)) ** 2) / 2.0 / tf.tanh(beta / 2.)
    dc_dphi = tf.reduce_sum(bigM * (dphi_idphi * (dc_drho * drho_dphi)), axis=0)
    dg_drhobar = 1.0 / alpha / nn * (1.0 / nn * tf.reduce_sum(rho_bar ** p)) ** (1.0 / p - 1) * rho_bar ** (p - 1)
    dg_dphi = tf.reduce_sum(bigM * (dphi_idphi * (tf.matmul(tf.cast(bigN, tf.float32),
                                                            (dg_drhobar / N_count.reshape(nn, 1))) * drho_dphi)), axis=0)
    #################################################################
    error_store.append(tf.reduce_sum((1-dg_dphi*(tf.reduce_sum(dc_dphi*dg_dphi)+1e-12)**(-1)*dg_dphi)**2))
    
    
error = tf.reduce_sum(error_store)    

global_step = tf.Variable(0, trainable=False)
starter_learning_rate = 0.0001
learning_rate = tf.train.exponential_decay(starter_learning_rate, global_step,
                                           1000, 1.0, staircase=True)

solver = tf.train.AdamOptimizer(learning_rate).minimize(error, global_step=global_step)

sess = tf.Session()
sess.run(tf.global_variables_initializer())

LHS = sio.loadmat('{}/LHS_train.mat'.format(directory_data))['LHS_train'] # pre-sampling the loading condition offline

LHS_x=np.int32(LHS[:,0])
LHS_y=np.int32(LHS[:,1])
LHS_z=LHS[:,2]

force=-1
F_batch = np.zeros([len(LHS), 2*(nelx+1)*(nely+1)])
error_store=[]
for i in range(len(LHS)):
    Fx = force * np.sin(LHS_z[i])
    Fy = force * np.cos(LHS_z[i])
    F_batch[i,2*((nely+1)*LHS_x[i]+LHS_y[i]+1)-1]=Fy
    F_batch[i,2*((nely+1)*LHS_x[i]+LHS_y[i]+1)-2]=Fx
    
F_load_input = LHS.copy()

#---------------------- start training -------------------
ratio=len(LHS)/batch_size
for epoch in range(10000):
    final_error=0
    for it in range(ratio):
        final_error_temp=sess.run(error,feed_dict={F:      F_batch[it%ratio*batch_size:it%ratio*batch_size+batch_size],
                                                   F_input:Fload_input[it%ratio*batch_size:it%ratio*batch_size+batch_size]})
        final_error=final_error + final_error_temp
    final_error=final_error/len(LHS)
    print('error is: {}'.format(final_error))
