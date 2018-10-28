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
batch_size = 3

'Input'
nelx, nely, alpha, alpha2, gamma, rmin, density_r = 12*4, 4*4, 0.6, 0.6, 3.0, 3.0, 6.0

'Algorithm parameters'
p, nn, epsilon_al, epsilon_opt = 16,nely * nelx, 1, 1e-3

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
# F = np.zeros([2 * (nely + 1) * (nelx + 1), 1])
# F[((nely + 1) * nelx) * 2 + nely + 1, 0] = -1
F = tf.placeholder(tf.float32, shape=([2*(nely+1)*(nelx+1),batch_size]))


# F=sps.coo_matrix(F)
U = np.zeros([2 * (nely + 1) * (nelx + 1), 1])

fixeddofs = np.asarray(np.arange(0, 2 * (nely + 1), 1).tolist())
fixeddofs = fixeddofs.reshape(1, len(fixeddofs))
alldofs = np.arange(2 * (nely + 1) * (nelx + 1)).reshape(1, 2 * (nely + 1) * (nelx + 1))

freedofs = np.setdiff1d(alldofs, fixeddofs)

'prepare some stuff to reduce cost'
dphi_idphi = (H / sum(H)).T

'Start Iteration'
z_dim, h_dim_1, h_dim_2, h_dim_3, h_dim_4, h_dim_5 = 2, 100, 100, 100, 100, 100

F_input = tf.placeholder(tf.float32, shape=([batch_size, z_dim]))
# F = tf.placeholder(tf.float32, shape=([2*(nely+1)*(nelx+1),batch_size]))

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

# phi = (tf.Variable(alpha * np.ones([nn, 1]), dtype='float32'))

loop = 0
# delta = 1.0

# while beta < 1000:
loop = loop + 1
loop2 = 0
'augmented lagrangian parameters'
# r_lag = 1
# r_lag2 = 0.001
# lamda_lag = 0
# lamda_lag2 = 0

eta = 0.1
eta2 = 0.1
epsilon = 1
dphi = 1e6

r_lag = tf.placeholder(tf.float32, shape=([batch_size,1]))
r_lag2 = tf.placeholder(tf.float32, shape=([batch_size,1]))
lamda_lag = tf.placeholder(tf.float32, shape=([batch_size,1]))
lamda_lag2 = tf.placeholder(tf.float32, shape=([batch_size,1]))
beta = tf.placeholder(tf.float32, shape=([batch_size,1]))
################################################################
'get initial g and c'

W1_fake = tf.Variable(xavier_init([z_dim, h_dim_1]), trainable=False)
b1_fake = tf.Variable(tf.zeros(shape=[h_dim_1]), trainable=False)
W1_fake = W1_fake.assign(W1)
b1_fake = b1_fake.assign(b1)

W2_fake = tf.Variable(xavier_init([h_dim_1, h_dim_2]), trainable=False)
b2_fake = tf.Variable(tf.zeros(shape=[h_dim_2]), trainable=False)
W2_fake = W2_fake.assign(W2)
b2_fake = b2_fake.assign(b2)

W3_fake = tf.Variable(xavier_init([h_dim_2, h_dim_3]), trainable=False)
b3_fake = tf.Variable(tf.zeros(shape=[h_dim_3]), trainable=False)
W3_fake = W3_fake.assign(W3)
b3_fake = b3_fake.assign(b3)

W4_fake = tf.Variable(xavier_init([h_dim_3, h_dim_4]), trainable=False)
b4_fake = tf.Variable(tf.zeros(shape=[h_dim_4]), trainable=False)
W4_fake = W4_fake.assign(W4)
b4_fake = b4_fake.assign(b4)

W5_fake = tf.Variable(xavier_init([h_dim_4, nn]), trainable=False)
b5_fake = tf.Variable(tf.zeros(shape=[nn]), trainable=False)
W5_fake = W5_fake.assign(W5)
b5_fake = b5_fake.assign(b5)

# W6_fake = tf.Variable(xavier_init([h_dim_5, nn]), trainable=False)
# b6_fake = tf.Variable(tf.zeros(shape=[nn]), trainable=False)
# W6_fake = W6_fake.assign(W6)
# b6_fake = b6_fake.assign(b6)

h1_fake = tf.contrib.layers.batch_norm(tf.nn.relu(tf.matmul(F_input, W1_fake) + b1_fake),scale=True,trainable=False)
h2_fake = tf.contrib.layers.batch_norm(tf.nn.relu(tf.matmul(h1_fake, W2_fake) + b2_fake),scale=True,trainable=False)
h3_fake = tf.contrib.layers.batch_norm(tf.nn.relu(tf.matmul(h2_fake, W3_fake) + b3_fake),scale=True,trainable=False)
h4_fake = tf.contrib.layers.batch_norm(tf.nn.relu(tf.matmul(h3_fake, W4_fake) + b4_fake),scale=True,trainable=False)
# h5_fake = tf.nn.relu(tf.matmul(h4_fake, W5_fake) + b5_fake)

phi_temp_fake = tf.sigmoid(tf.matmul(h4_fake, W5_fake) + b5_fake)
phi_fake = tf.reshape(phi_temp_fake,[nn,batch_size])


##################################################################################################
sep_grad_store,error_store,dphi_store,dobj_store=[],[],[],[]
c,g,global_density=[],[],[]
c_fake,g_fake,global_density_fake=[],[],[]
rho,rho_fake=[],[]
dphi_fake=[]

tic=time.clock()

learning_rate_fake = tf.placeholder(tf.float32,shape=([batch_size,1]))
g_old = [tf.Variable(0, dtype='float32',trainable=False)]*batch_size
global_density_old = [tf.Variable([[0]], dtype='float32',trainable=False)]*batch_size
c_old = [tf.Variable(0, dtype='float32',trainable=False)]*batch_size
tic=time.clock()

for i in range(batch_size):
    phi_til = tf.matmul(tf.cast(H, tf.float32), phi[:,i:i+1]) / Hs.reshape(nn, 1)
    rho.append((tf.tanh(beta[i][0] / 2.0) + tf.tanh(beta[i][0] * (phi_til - 0.5))) / (2 * tf.tanh(beta[i][0] / 2.0)))

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

    # if ((max(abs(dphi)))<epsilon_al and g<0.005 and loo2>2) or (loo2>100):


    dc_drho = -gamma * rho[i] ** (gamma - 1) * (E0 - Emin) * ce
    drho_dphi = beta[i][0] * (1 - tf.tanh(beta[i][0] * (phi[:,i:i+1] - 0.5)) ** 2) / 2.0 / tf.tanh(beta[i][0] / 2.)
    dc_dphi = tf.reduce_sum(bigM * (dphi_idphi * (dc_drho * drho_dphi)), axis=0)
    dg_drhobar = 1.0 / alpha / nn * (1.0 / nn * tf.reduce_sum(rho_bar ** p)) ** (1.0 / p - 1) * rho_bar ** (p - 1)
    dg_dphi = tf.reduce_sum(bigM * (dphi_idphi * (tf.matmul(tf.cast(bigN, tf.float32),
                                                            (dg_drhobar / N_count.reshape(nn, 1))) * drho_dphi)), axis=0)

    dphi = dc_dphi + tf.cast(lamda_lag[i][0] * dg_dphi * tf.reduce_sum(tf.cast(g[i] > 0, tf.float32)), tf.float32) + \
           tf.cast(2.0 / r_lag[i][0] * g[i] * dg_dphi * tf.reduce_sum(tf.cast(g[i] > 0, tf.float32)), tf.float32) + \
           lamda_lag2[i][0] * tf.ones(nn, 1) / nn * (tf.reduce_sum(tf.cast(global_density[i] > 0, tf.float32))) + \
           tf.reshape(2.0 / r_lag2[i][0] * global_density[i] / nn / nn * tf.ones([nn, 1]) * (
           tf.reduce_sum(tf.cast(global_density[i] > 0, tf.float32))), [-1])
    
    dphi_store.append(dphi)
    

    
############################################################################################################

    'get initial g and c'
    phi_til_fake = tf.matmul(tf.cast(H, tf.float32), phi_fake[:,i:i+1]) / Hs.reshape(nn, 1)
    rho_fake.append((tf.tanh(beta[i][0] / 2.0) + tf.tanh(beta[i][0] * (phi_til_fake - 0.5))) / (2 * tf.tanh(beta[i][0] / 2.0)))

    sK_temp_fake = tf.transpose((KE.reshape(KE.shape[0] * KE.shape[1], 1) * \
                                 tf.reshape(Emin + tf.reshape(rho_fake[i], [-1]) ** (gamma) * (E0 - Emin), [1, nelx * nely])))
    sK_fake = tf.reshape(sK_temp_fake, [8 * 8 * nelx * nely, 1])

    ###################
    indices_m = np.stack((iK.reshape(-1), jK.reshape(-1)), axis=1)
    values_m_fake = tf.reshape(sK_fake, [-1])

    linearized_m = tf.matmul(indices_m, [[10000], [1]])
    y_m, idx_m = tf.unique(tf.squeeze(linearized_m))

    idx_m_sort, ind_m_sort = tf.nn.top_k(idx_m, k=nelx * nely * 64)
    idx_m_sort = tf.reverse(idx_m_sort, [0])
    ind_m_sort = tf.reverse(ind_m_sort, [0])

    values_m_fake = tf.gather(values_m_fake, ind_m_sort)
    values_m_fake = tf.segment_sum(values_m_fake, idx_m_sort)

    y_m = tf.expand_dims(y_m, 1)
    indices_m = tf.concat([y_m // 10000, y_m % 10000], axis=1)
    #####################

    #####################
    K_sp_fake = tf.SparseTensor(tf.cast(indices_m, tf.int64),
                                tf.reshape(tf.cast(values_m_fake, tf.float32), [-1]),
                                [(nely + 1) * (nelx + 1) * 2, (nely + 1) * (nelx + 1) * 2])
    K_sp_fake = tf.sparse_add(tf.zeros(((nely + 1) * (nelx + 1) * 2, (nely + 1) * (nelx + 1) * 2)), K_sp_fake)
    K_dense_fake = (K_sp_fake + tf.transpose(K_sp_fake)) / 2.0
    #####################

    K_temp_fake = tf.gather(K_dense_fake, freedofs.astype(np.int32))
    K_free_fake = tf.transpose(tf.gather(tf.transpose(K_temp_fake), freedofs.astype(np.int32)))

    F_RHS_fake=tf.cast(tf.gather(tf.reshape(F[:,i:i+1], [(nelx + 1) * (nely + 1) * 2, 1]), freedofs.astype(np.int64)),
                              tf.float32)
    chol_fake = tf.cholesky(K_free_fake+np.diag((np.ones([1,(nelx+1)*(nely+1)*2-len(fixeddofs[0])])*1e-8).tolist()[0]))
    U_pre_fake = tf.cholesky_solve(chol_fake,F_RHS_fake)
    
#     U_pre_fake = tf.matmul(tf.matrix_inverse(tf.cast(K_free_fake, tf.float32)),
#                            tf.cast(tf.gather(tf.reshape(F[:,i:i+1], [(nelx + 1) * (nely + 1) * 2, 1]), freedofs.astype(np.int64)),
#                                    tf.float32))

    U_fake = tf.sparse_add(tf.zeros(((nelx + 1) * (nely + 1) * 2)),
                           tf.SparseTensor(freedofs.reshape((nelx + 1) * (nely + 1) * 2 - (nely + 1) * 2, 1),
                                           tf.reshape(tf.cast(U_pre_fake, tf.float32), [-1]),
                                           [(nelx + 1) * (nely + 1) * 2]))

    'Objective Function and Sensitivity Analysis'
    ce_fake = tf.reduce_sum(tf.multiply(tf.matmul(tf.reshape(tf.gather(U_fake, edofMat.astype(np.int32)), edofMat.shape),
                                                  KE.astype(np.float32)), tf.gather(U_fake, edofMat.astype(np.int32))),
                            axis=1)
    ce_fake = tf.reshape(ce_fake, [nn, 1])
    c_fake.append(tf.reduce_sum(tf.multiply(tf.cast(tf.multiply(rho_fake[i] ** gamma, (E0 - Emin)), tf.float32), ce_fake)))
    rho_bar_fake=tf.divide(tf.matmul(tf.cast(bigN, tf.float32), rho_fake[i]), N_count.reshape(nn, 1))
    g_fake.append((tf.reduce_sum(rho_bar_fake ** p) / nely / nelx) ** (1. / p) / alpha - 1.0)
    global_density_fake.append(tf.matmul(tf.transpose(rho_fake[i]), tf.ones([nn, 1])) / nn - alpha2)
    ###################################################################

    # if ((max(abs(dphi)))<epsilon_al and g<0.005 and loo2>2) or (loo2>100):


    dc_drho_fake = -gamma * rho_fake[i] ** (gamma - 1) * (E0 - Emin) * ce_fake
    drho_dphi_fake = beta[i][0] * (1 - tf.tanh(beta[i][0] * (phi_fake[:,i:i+1] - 0.5)) ** 2) / 2.0 / tf.tanh(beta[i][0] / 2.)
    dc_dphi_fake = tf.reduce_sum(bigM * (dphi_idphi * (dc_drho_fake * drho_dphi_fake)), axis=0)
    dg_drhobar_fake = 1.0 / alpha / nn * (1.0 / nn * tf.reduce_sum(rho_bar_fake ** p)) ** (1.0 / p - 1) * rho_bar_fake ** (
    p - 1)
    dg_dphi_fake = tf.reduce_sum(bigM * (dphi_idphi * (tf.matmul(tf.cast(bigN, tf.float32),
                                                                 (dg_drhobar_fake / N_count.reshape(nn,
                                                                                                    1))) * drho_dphi_fake)),
                                 axis=0)

    A = dc_dphi_fake + tf.cast(lamda_lag[i][0] * dg_dphi_fake * tf.reduce_sum(tf.cast(g_fake[i] > 0, tf.float32)), tf.float32) + \
        tf.cast(2.0 / r_lag[i][0] * g_fake[i] * dg_dphi_fake * tf.reduce_sum(tf.cast(g_fake[i] > 0, tf.float32)), tf.float32)

    B = lamda_lag2[i][0] * tf.ones(nn, 1) / nn * (tf.reduce_sum(tf.cast(global_density_fake[i] > 0, tf.float32))) + \
        2.0 / r_lag2[i][0] * global_density_fake[i] / nn / nn * tf.ones([nn]) * (tf.reduce_sum(tf.cast(global_density_fake[i] > 0, tf.float32)))

    dphi_fake.append(A + B)
    
    
    error_sep = tf.reduce_sum(((tf.reshape(dphi_fake[i], [nn, 1]) * phi[:,i:i+1])))
    error_store.append(error_sep)
    
    sep_grad=tf.reduce_sum(tf.abs(tf.gradients((tf.reshape(dphi_fake[i], [nn, 1]) * phi[:,i:i+1]),W1)))
    sep_grad_store.append(sep_grad)
    
############################################################################################
#     g_old = tf.Variable(0, dtype='float32',trainable=False)
#     global_density_old = tf.Variable([[0]], dtype='float32',trainable=False)
#     c_old = tf.Variable(0, dtype='float32',trainable=False)

    g_old[i] = g_old[i].assign(g[i])
    global_density_old[i] = global_density_old[i].assign(global_density[i])
    c_old[i] = c_old[i].assign(c[i])

#     assign_g=tf.assign(g_old[i],g[i])
#     assign_gd=tf.assign(global_density_old[i],global_density[i])
#     assign_c=tf.assign(c_old[i],c[i])

    'learning rate adjusting'
    delta = -dphi_fake[i] * learning_rate_fake[i][0]
    phi_temp = phi_fake[:,i:i+1] + tf.reshape(delta, [nn, 1])

    # assign_phi = tf.assign(phi,phi_temp)

    # phi_temp=tf.Variable(np.ones([nn,1]), dtype='float32')
    # phi_temp=phi_temp.assign(phi)

    phi_til_temp = tf.matmul(tf.cast(H, tf.float32), phi_temp) / tf.reshape(tf.cast(Hs, tf.float32), [nn, 1])
    rho_temp = (tf.tanh(beta[i][0] / 2.0) + tf.tanh(beta[i][0] * (phi_til_temp - 0.5))) / (2 * tf.tanh(beta[i][0] / 2.0))
    rho_bar_temp = (tf.matmul(tf.cast(bigN, tf.float32), rho_temp) / tf.reshape(tf.cast(N_count, tf.float32), [nn, 1]))
    g_temp = (tf.reduce_sum(rho_bar_temp ** p) / nn) ** (1.0 / p) / alpha - 1.0
    global_density_temp = tf.matmul(tf.transpose(rho_temp), tf.ones([nn, 1])) / nn - alpha2

    sK_temp_fake2 = tf.transpose((KE.reshape(KE.shape[0] * KE.shape[1], 1) * \
                                  tf.reshape(Emin + tf.reshape(rho_temp, [-1]) ** (gamma) * (E0 - Emin), [1, nelx * nely])))
    sK_fake2 = tf.reshape(sK_temp_fake2, [8 * 8 * nelx * nely, 1])

    ###################
    values_m_fake2 = tf.reshape(sK_fake2, [-1])
    values_m_fake2 = tf.gather(values_m_fake2, ind_m_sort)
    values_m_fake2 = tf.segment_sum(values_m_fake2, idx_m_sort)
    ###################

    ###################
    K_sp_fake2 = tf.SparseTensor(tf.cast(indices_m, tf.int64),
                                 tf.reshape(tf.cast(values_m_fake2, tf.float32), [-1]),
                                 [(nely + 1) * (nelx + 1) * 2, (nely + 1) * (nelx + 1) * 2])
    K_sp_fake2 = tf.sparse_add(tf.zeros(((nely + 1) * (nelx + 1) * 2, (nely + 1) * (nelx + 1) * 2)), K_sp_fake2)
    K_dense_fake2 = (K_sp_fake2 + tf.transpose(K_sp_fake2)) / 2.0
    ###################

    K_temp_fake2 = tf.gather(K_dense_fake2, freedofs.astype(np.int32))
    K_free_fake2 = tf.transpose(tf.gather(tf.transpose(K_temp_fake2), freedofs.astype(np.int32)))
    
    F_RHS_fake2=tf.cast(tf.gather(tf.reshape(F[:,i:i+1], [(nelx + 1) * (nely + 1) * 2, 1]), freedofs.astype(np.int64)),
                              tf.float32)
    chol_fake2 = tf.cholesky(K_free_fake2+np.diag((np.ones([1,(nelx+1)*(nely+1)*2-len(fixeddofs[0])])*1e-8).tolist()[0]))
    U_pre_fake2 = tf.cholesky_solve(chol_fake2,F_RHS_fake2)
    
#     U_pre_fake2 = tf.matmul(tf.matrix_inverse(tf.cast(K_free_fake2, tf.float32)+np.diag((np.ones([1,(nelx+1)*(nely+1)*2-len(fixeddofs[0])])*1e-15).tolist()[0])),
#                             tf.cast(tf.gather(tf.reshape(F[:,i:i+1], [(nelx + 1) * (nely + 1) * 2, 1]), freedofs.astype(np.int64)),tf.float32))

    U_fake2 = tf.sparse_add(tf.zeros(((nelx + 1) * (nely + 1) * 2)),
                            tf.SparseTensor(freedofs.reshape((nelx + 1) * (nely + 1) * 2 - (nely + 1) * 2, 1),
                                            tf.reshape(tf.cast(U_pre_fake2, tf.float32), [-1]),
                                            [(nelx + 1) * (nely + 1) * 2]))

    ce_temp = tf.reduce_sum(tf.multiply(tf.matmul(tf.reshape(tf.gather(U_fake2, edofMat.astype(np.int32)), edofMat.shape),
              KE.astype(np.float32)), tf.gather(U_fake2, edofMat.astype(np.int32))),axis=1)
    
    ce_temp = tf.reshape(ce_temp, [nn, 1])
    c_temp = tf.reduce_sum(tf.multiply(tf.cast(tf.multiply(rho_temp ** gamma, (E0 - Emin)), tf.float32), ce_temp))

    A2 = (c_temp + lamda_lag[i][0] * g_temp * (tf.reduce_sum(tf.cast(g_temp > 0, tf.float32))) + 1.0 / r_lag[i][0] * g_temp ** 2 * (
    tf.reduce_sum(tf.cast(g_temp > 0, tf.float32))) + \
          1.0 / r_lag2[i][0] * global_density_temp ** 2 * (tf.reduce_sum(tf.cast(global_density_temp > 0, tf.float32))))

    B2 = (c_old[i] + lamda_lag[i][0] * g_old[i] * (tf.reduce_sum(tf.cast(g_old[i] > 0, tf.float32))) + 1.0 / r_lag[i][0] * g_old[i] ** 2 * (
    tf.reduce_sum(tf.cast(g_old[i] > 0, tf.float32))) + \
          1.0 / r_lag2[i][0] * global_density_old[i] ** 2 * (tf.reduce_sum(tf.cast(global_density_old[i] > 0, tf.float32))))
    dobj = A2 - B2
    dobj_store.append(dobj)
    
toc=time.clock()
print(toc-tic)


#######################################
global_step = tf.Variable(0, trainable=False)
starter_learning_rate = 0.001
learning_rate = tf.train.exponential_decay(starter_learning_rate, global_step,
                                           1000, 1.0, staircase=True)

solver0 = tf.train.AdamOptimizer(learning_rate).minimize(error_store[0], global_step=global_step)
solver1 = tf.train.AdamOptimizer(learning_rate).minimize(error_store[1], global_step=global_step)
solver2 = tf.train.AdamOptimizer(learning_rate).minimize(error_store[2], global_step=global_step)

sess = tf.Session()
sess.run(tf.global_variables_initializer())

#######################################
force=-1
F_batch=np.zeros([batch_size,z_dim])

# alpha=1.5
theta = np.linspace(np.pi*0.,2.0*np.pi/5.,batch_size)

count=0
for i in range(batch_size-1,-1,-1):
#     theta = 2*(i+1)*np.pi/10.#+i*np.pi/9*alpha
    print theta[i]
#     theta = (i+1)*np.pi/10
    Fx = force*np.sin(theta[i])
    Fy = force*np.cos(theta[i])
    
    # up-right corner
    F_batch[count,0]=Fx
    F_batch[count,1]=Fy
    count=count+1   

count=0
F_sp=np.zeros([2*(nely+1)*(nelx+1),batch_size])
for i in range(batch_size-1,-1,-1):
#     theta = 2*(i+1)*np.pi/10.#+i*np.pi/9*alpha
#     theta = (i+1)*np.pi/5
    Fx = force*np.sin(theta[i])
    Fy = force*np.cos(theta[i])    
    F_sp[(nely+1)*2*nelx+nely, count] = Fx
    F_sp[(nely+1)*2*nelx+nely+1, count] = Fy
    count=count+1
    
lamda_value = np.zeros([batch_size, 1])
lamda_value2 = np.zeros([batch_size, 1])
r_value = np.ones([batch_size, 1]) * 1
r_value2 = np.ones([batch_size, 1]) * 0.001
eta = np.ones([batch_size, 1]) * 0.1
eta2 = np.ones([batch_size, 1]) * 0.1
j, j2 = 0, 0

k_j = np.ones([batch_size, 1])
thre_range = 0.02
agg = 0
total_E_pre = 1e20
count_k = 0
count = 0
drho=1e6
it=0
beta_value=np.ones([batch_size,1])*10
learning_rate_fake_value=np.ones([batch_size,1])*0.001


import os
import time
timestr = time.strftime("%Y%m%d-%H%M%S_bn")
if not os.path.exists(timestr):
    os.makedirs(timestr)
    

##################################################
it=0
drho_total_store=[]

while min(beta_value)<11:
    it=it+1
  
    
    loop2=0
        
    while 1:
#         try:
        g_value_temp,dphi_fake_value,rho_pre = sess.run([g,dphi_fake,rho], 
                                              feed_dict={learning_rate_fake: learning_rate_fake_value,
                                                         lamda_lag:lamda_value,lamda_lag2:lamda_value2,
                                                         r_lag:r_value,r_lag2:r_value2,F_input: F_batch,
                                                         F:F_sp,learning_rate:starter_learning_rate,
                                                         beta:beta_value})
#         except:
#             stop=1

        
        dphi_fake_value_max=max(max(abs(np.array(dphi_fake_value[0])).reshape(-1)),
                                max(abs(np.array(dphi_fake_value[1])).reshape(-1)),
                                max(abs(np.array(dphi_fake_value[2])).reshape(-1)))
        
        
        if (dphi_fake_value_max < 1*batch_size and max(g_value_temp) < 0.01 and loop2 > 0) or (loop2 > 100):

#         if (drho < 0.05 and max(g_value_temp) < 0.005 and loop2 > 0) or (loop2 > 100):
            #             print('wrong')
            lamda_value = np.zeros([batch_size, 1])
            lamda_value2 = np.zeros([batch_size, 1])
            r_value = np.ones([batch_size, 1]) * 1
            r_value2 = np.ones([batch_size, 1]) * 0.001
            eta = np.ones([batch_size, 1]) * 0.1
            eta2 = np.ones([batch_size, 1]) * 0.1
            dphi_fake_value=np.array([1e6]*batch_size)
            drho=np.array([1e6]*batch_size)
            break

        loop2 = loop2 + 1
        j, j2 = 0, 0
        loop3 = np.array([0]*batch_size)
        dphi_fake_value = np.array([1e6]*batch_size)
        drho = np.array([1e6]*batch_size)
    


        while (max(max(abs(np.array(dphi_fake_value[0])).reshape(-1)),
                   max(abs(np.array(dphi_fake_value[1])).reshape(-1)),
                   max(abs(np.array(dphi_fake_value[2])).reshape(-1)))) > epsilon and min(loop3) < 1000:
            
#             rho_value_pre = sess.run(rho, feed_dict={learning_rate_fake: learning_rate_fake_value,lamda_lag:lamda_value,lamda_lag2:lamda_value2,r_lag:r_value,r_lag2:r_value2,
#                                  F_input: F_batch,F:F_sp,learning_rate:starter_learning_rate,beta:beta_value})

            
            for i in range(batch_size):
                if count>=4000:
                    break
                if loop3[i]>1000:
                    continue
#                 dphi_fake_value[i] = np.array([1e6])
#                 g_value = g_value_temp[i]

                starter_learning_rate = 0.001
                learning_rate_fake_value[i]=0.001
                loop3[i] = loop3[i] + 1
                loop4 = np.asarray([0]*batch_size)

                tic=time.clock()
                while loop4[i] < 10:
                    try:
                        dphi_fake_value, dobj_value,rho_value = sess.run([dphi_fake, dobj_store,rho],
                                                   feed_dict={learning_rate_fake: learning_rate_fake_value,lamda_lag:lamda_value,
                                                            lamda_lag2:lamda_value2, r_lag:r_value,
                                                            r_lag2:r_value2,F_input:F_batch,
                                                            F:F_sp,learning_rate:learning_rate_fake_value[i][0],
                                                            beta:beta_value})
                    except:
                        starter_learning_rate = starter_learning_rate*0.5
                        learning_rate_fake_value[i] = learning_rate_fake_value[i]*0.5
                        


                    if dobj_value[i] > 0:
                        starter_learning_rate = starter_learning_rate * 0.5
                        learning_rate_fake_value[i] = learning_rate_fake_value[i]*0.5
                        loop4[i] = loop4[i] + 1
                        if loop4[i] == 10:
                            loop3[i] = 10000
                            dphi_fake_value[i] = np.array([0])
                            drho[i] = np.array([0])
                            #                         break
                    else:
                        if i == 0:
                            try:
                                sess.run([solver0],feed_dict={learning_rate_fake: learning_rate_fake_value,lamda_lag:lamda_value,
                                                           lamda_lag2:lamda_value2,r_lag:r_value,r_lag2:r_value2,
                                                           F_input: F_batch,F:F_sp,learning_rate:learning_rate_fake_value[i][0],beta:beta_value})

                            except:
                                print('jump0')
                                
                        if i == 1:
                            try:
                                sess.run([solver1],feed_dict={learning_rate_fake: learning_rate_fake_value,lamda_lag:lamda_value,
                                                           lamda_lag2:lamda_value2,r_lag:r_value,r_lag2:r_value2,
                                                           F_input: F_batch,F:F_sp,learning_rate:learning_rate_fake_value[i][0],beta:beta_value})

                            except:
                                print('jump1')
                                
                        if i == 2:
                            try:
                                sess.run([solver2],feed_dict={learning_rate_fake: learning_rate_fake_value,lamda_lag:lamda_value,
                                                           lamda_lag2:lamda_value2,r_lag:r_value,r_lag2:r_value2,
                                                           F_input: F_batch,F:F_sp,learning_rate:learning_rate_fake_value[i][0],beta:beta_value})
                            except:
                                print('jump2')
                            
                        break

                toc=time.clock()
                count = count + 1
#                 try:
                g_value,c_value,sep_grad_value,error_total_value,phi_value,rho_curr=sess.run([g,c,sep_grad_store,error_store,phi,rho], 
                    feed_dict={learning_rate_fake: learning_rate_fake_value, lamda_lag: lamda_value,
                    lamda_lag2: lamda_value2,r_lag: r_value,
                    r_lag2: r_value2, F_input: F_batch, F: F_sp, learning_rate: learning_rate_fake_value[i][0],
                    beta: beta_value})
#                 except:
#                     print('jump3')

                drho[i]=sum(abs(rho_curr[i]-rho_pre[i]).reshape(-1))
                drho_total=sum(abs(np.array(rho_curr)-np.array(rho_pre)).reshape(-1))
                drho_total_store.append(drho_total)
                print('iteration:{}'.format(it))
                print('updating_case:{}'.format(i))
                print('num_update:{}'.format(count))
                print('total_error:{}'.format(np.sum(np.array(error_total_value))))
                print('g_value:{}'.format(g_value))
                print('c_value:{}'.format(c_value))
                print('log_r:{}'.format(np.log(r_value)))
#                 print('max_abs_dphi:{}'.format(max(abs(dphi_fake_value.reshape(-1)))))
                print('gradient_W1:{}'.format(sep_grad_value))
                print('dobj_value:{}'.format(dobj_value))
                #                                                                     r_lag2:r_value2,F_input: F_batch,F:F_sp,learning_rate:starter_learning_rate,beta:beta_value})))
                print('dphi_fake_value_max:{}'.format((max(abs(dphi_fake_value[i].reshape(-1))))))
#                 print('drho_fake_value_max:{}'.format(max(abs(rho_value[i].reshape(-1)-rho_value_pre[i].reshape(-1)))))
                print('drho:{}'.format(drho))
                print('loop3:{}'.format(loop3))
                print
                print('running time:{}'.format(toc-tic))
                print()
                if count%500 == 0:
                    sio.savemat('{}/c_value.mat'.format(timestr), mdict={'c': np.array(c_value)})
                    sio.savemat('{}/rho.mat'.format(timestr), mdict={'rho': np.array(rho_curr)})
                    sio.savemat('{}/drho.mat'.format(timestr), mdict={'drho': np.array(drho_total_store)})
                    fig, axs = plt.subplots(2, 5, figsize=(15, 4), facecolor='w', edgecolor='k')
                    fig.subplots_adjust(hspace=.1, wspace=.001)

                    axs = axs.ravel()

                    for i in range(batch_size):
                        axs[i].imshow(1 - rho_curr[i].reshape([nelx, nely]).T, 'gray')
                        axs[i].set_title('c={}'.format(c_value[i]))
                    plt.savefig('{}/opt_{}.png'.format(timestr, str(count)))
                    plt.close()
                if count >= 4000:
                    break
#             drho1=max(rho_value_cur[0].reshape(-1)-rho_value_pre[0].reshape(-1))
#             drho2=max(rho_value_cur[1].reshape(-1)-rho_value_pre[1].reshape(-1))
#             drho3=max(rho_value_cur[2].reshape(-1)-rho_value_pre[2].reshape(-1))
#             drho=max(drho1,drho2,drho3)
        
    
#         g_value, gd, rho_value_cur = sess.run([g, global_density,rho], 
#                                      feed_dict={learning_rate_fake: learning_rate_fake_value,lamda_lag:lamda_value,
#                                      lamda_lag2:lamda_value2,r_lag:r_value,
#                                      r_lag2:r_value2,F_input: F_batch,F:F_sp,
#                                      learning_rate:starter_learning_rate,beta:beta_value})
        for i in range(batch_size):    
            g_value, gd= sess.run([g, global_density], 
                                     feed_dict={learning_rate_fake: learning_rate_fake_value,lamda_lag:lamda_value,
                                     lamda_lag2:lamda_value2,r_lag:r_value,
                                     r_lag2:r_value2,F_input: F_batch,F:F_sp,
                                     learning_rate:learning_rate_fake_value[i][0],beta:beta_value})            
            if g_value[i] < eta[i]:
                lamda_value[i] = lamda_value[i] + 2 * g_value[i] / r_value[i]
                j = j + 1
                eta[i] = eta[i] * 0.5
            else:
                r_value[i] = r_value[i] * 0.5
                j = 0

            if gd[i] < eta2[i]:
                lamda_value2[i] = lamda_value2[i] + 2 * gd[i][0] / r_value2[i]
                j2 = j2 + 1
                eta2[i] = eta2[i] * 0.5
            else:
                r_value2[i] = r_value2[i] * 0.5
                j2 = 0
                
    for i in range(batch_size):
        beta_value[i] = 1.5 * beta_value[i]

phi_value, rho_value = sess.run([phi, rho],feed_dict={learning_rate_fake: learning_rate_fake_value, lamda_lag: lamda_value,lamda_lag2: lamda_value2,r_lag: r_value,
               r_lag2: r_value2, F_input: F_batch, F: F_sp, learning_rate: starter_learning_rate, beta: beta_value})

print('finished')