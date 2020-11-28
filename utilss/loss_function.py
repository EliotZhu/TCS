# loss_function.py
#
# Author: Elliott Jie Zhu
# Tested with Python version 3.7 and Keras version 2 (using TensorFlow backend)

import tensorflow as tf
from tensorflow.keras import backend as K


def surv_likelihood_lrnn(window_size, alpha, beta, gamma, mask_value = -10):
    def loss(y_true, y_pred):

        def partial_loss(y_true_temp, y_pred_temp):
            mask = K.equal(y_true_temp[:, 0:window_size], mask_value)
            mask = 1 - K.cast(mask, K.floatx())
            cens_uncens = 1. + (y_pred_temp - 1.) * y_true_temp[:, 0:window_size]* mask  # component for all patients
            uncens = 1. - y_pred_temp * y_true_temp[:, window_size:2 * window_size]* mask   # component for uncensored patients
            loss_1 = K.sum(-K.log(K.clip(K.concatenate( (cens_uncens, uncens) ), K.epsilon(), None)),-1)  # return -log likelihood
            return loss_1

        def debias_loss(y_true_temp, y_pred_temp, propensity_temp, D = 1.0):
            mask = K.equal(y_true_temp[:, 0:window_size], mask_value)
            mask = 1 - K.cast(mask, K.floatx())

            mse = K.mean((y_true_temp[:, 0:window_size]*mask - y_pred_temp*mask )**2 * propensity_temp)
            loss = (mse) /K.mean(propensity_temp * (1.0-propensity_temp))
            return loss

        def rank_loss(y_true_temp, y_pred_temp):
            mask = K.equal(y_true_temp[:, 0:window_size], mask_value)
            mask = 1 - K.cast(mask, K.floatx())
            # Rank loss
            R = K.dot(y_pred_temp*mask, K.transpose(y_true_temp[:, 0:window_size]*mask ))  # N*N
            diag_R = K.reshape(tf.linalg.diag_part(R), (-1, 1))  # N*1
            one_vector = tf.ones(K.shape(diag_R))  # N*1
            R = K.dot(one_vector, K.transpose(diag_R)) - R  # r_{i}(T_{j}) - r_{j}(T_{j})
            R = K.transpose(R)  # r_{i}(T_{i}) - r_{j}(T_{i})

            I2 = K.reshape(K.sum(y_true_temp[:, window_size:2 * window_size]*mask, axis=1), (-1, 1))  # N*1
            I2 = K.cast(K.equal(I2, 1), dtype=tf.float32)
            I2 = K.dot(one_vector, K.reshape(I2, (1, -1)))
            T2 = K.reshape(K.sum(y_true_temp[:, 0:window_size]*mask, axis=1), (-1, 1))  # N*1

            T = K.relu( K.sign(K.dot(one_vector, K.transpose(T2)) - K.dot(T2, K.transpose(one_vector))))  # (Ti(t)>Tj(t)=1); N*N
            T = I2 * T  # only remains T_{ij}=1 when event occured for subject i 1*N

            eta = T * K.exp(-R / 0.1)
            eta = T * K.cast(eta >= T, dtype=tf.float32)
            loss_2 = 1 - K.sum(eta) / (1 + K.sum(T))

            return  loss_2 #+ brier_score

        propensity = y_pred[:, 2, :]
        y_pred1 = y_pred[:, 0, :]
        y_pred0 = y_pred[:, 1, :]

        propensity_1 = 1.0 - K.clip(propensity, 0.0001, 1.0)


        partial_loss1 = partial_loss(y_true[:, 0:window_size*2], y_pred1)
        partial_loss2 = partial_loss(y_true[:, window_size*2:window_size*4], y_pred0)


        rank_loss1 = rank_loss(y_true[:, window_size*2:window_size*4], y_pred0)
        rank_loss2 = rank_loss(y_true[:, 0:window_size*2], y_pred1)


        debias_loss1 = debias_loss(y_true[:, 0:window_size*2], y_pred1, propensity_1)
        debias_loss2 = debias_loss(y_true[:, window_size*2:window_size*4], y_pred0, propensity_1)
        causal_loss3 = (debias_loss1+debias_loss2)/2 #+ K.abs(K.mean(y_pred0[y_true[:, -1] == 1]) - K.mean(y_pred0[y_true[:, -1] == 0]))


        return alpha * (partial_loss1+partial_loss2) + beta * (rank_loss1+rank_loss2)  + causal_loss3 * gamma

    return loss



def prop_likelihood_lrnn(window_size):
    def loss(y_true, y_pred):
        propensity = y_pred,
        propensity_loss =  K.mean((tf.repeat(K.reshape(y_true[:, -1], (-1,1)) , window_size, axis=1) - propensity)**2)
        return propensity_loss
    return loss



