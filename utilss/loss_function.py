# loss_function.py
#
# Author: Elliott Jie Zhu
# Tested with Python version 3.7 and Keras version 2 (using TensorFlow backend)

import tensorflow as tf
from tensorflow.keras import backend as K


def surv_likelihood_lrnn(window_size, alpha, beta, gamma, mask_value = -10):
    def loss(y_true, y_pred):
        def partial_loss(y_true_1, y_true_0, y_pred_1, y_pred_0):
            mask1 = K.equal(y_true_1[:, 0:window_size], mask_value)
            mask1 = 1 - K.cast(mask1, K.floatx())
            mask0 = K.equal(y_true_0[:, 0:window_size], mask_value)
            mask0 = 1 - K.cast(mask0, K.floatx())

            y_pred = y_pred_1 * mask1 + y_pred_0 * mask0
            y_true_p1 = y_true_1[:, 0:window_size] * mask1 + y_true_0[:, 0:window_size] * mask0
            y_true_p2 = y_true_1[:, window_size:2 * window_size] * mask1 + \
                        y_true_0[:, window_size:2 * window_size] * mask0

            cens_uncens = 1. + (y_pred - 1.) * y_true_p1  # component for all patients
            uncens = 1. - y_pred * y_true_p2  # component for uncensored patients
            loss = K.sum(-K.log(K.clip(K.concatenate((cens_uncens, uncens)), K.epsilon(), None)),
                         -1)  # return -log likelihood

            return loss

        def debias_loss(y_true_1, y_true_0, y_pred_1, y_pred_0, propensity_temp, D=1.0):
            # mask1 = K.equal(y_true_1[:, 0:window_size], mask_value)
            # mask1 = 1 - K.cast(mask1, K.floatx())
            #
            # mask0 = K.equal(y_true_0[:, 0:window_size], mask_value)
            # mask0 = 1 - K.cast(mask0, K.floatx())
            #
            # loss1 = K.mean((y_pred_1 * mask1 - y_pred_0 * mask1) ** 2)
            # loss2 = K.mean((y_pred_1 * mask0 - y_pred_0 * mask0) ** 2)
            mask1 = K.equal(y_true_1[:, 0:window_size], mask_value)
            mask1 = 1 - K.cast(mask1, K.floatx())
            mask0 = K.equal(y_true_0[:, 0:window_size], mask_value)
            mask0 = 1 - K.cast(mask0, K.floatx())

            y_pred = y_pred_1 * mask1 + y_pred_0 * mask0
            y_true_p1 = y_true_1[:, 0:window_size] * mask1 + y_true_0[:, 0:window_size] * mask0
            y_true_p2 = y_true_1[:, window_size:2 * window_size] * mask1 + \
                        y_true_0[:, window_size:2 * window_size] * mask0

            loss = K.mean((y_pred_1 * mask1 - y_pred_0 * mask0)**2)# * propensity_temp)
            # /K.mean(propensity_temp * (1.0-propensity_temp))

            return loss

        def rank_loss(y_true_1, y_true_0, y_pred_1, y_pred_0):
            mask1 = K.equal(y_true_1[:, 0:window_size], mask_value)
            mask1 = 1 - K.cast(mask1, K.floatx())
            mask0 = K.equal(y_true_0[:, 0:window_size], mask_value)
            mask0 = 1 - K.cast(mask0, K.floatx())

            y_pred = y_pred_1 * mask1 + y_pred_0 * mask0
            y_true_p1 = y_true_1[:, 0:window_size] * mask1 + y_true_0[:, 0:window_size] * mask0
            y_true_p2 = y_true_1[:, window_size:2 * window_size] * mask1 + \
                        y_true_0[:, window_size:2 * window_size] * mask0

            # Rank loss
            R = K.dot(y_pred, K.transpose(y_true_p1))  # N*N
            diag_R = K.reshape(tf.linalg.diag_part(R), (-1, 1))  # N*1
            one_vector = tf.ones(K.shape(diag_R))  # N*1
            R = K.dot(one_vector, K.transpose(diag_R)) - R  # r_{i}(T_{j}) - r_{j}(T_{j})
            R = K.transpose(R)  # r_{i}(T_{i}) - r_{j}(T_{i})

            I2 = K.reshape(K.sum(y_true_p2, axis=1), (-1, 1))  # N*1
            I2 = K.cast(K.equal(I2, 1), dtype=tf.float32)
            I2 = K.dot(one_vector, K.reshape(I2, (1, -1)))
            T2 = K.reshape(K.sum(y_true_p1, axis=1), (-1, 1))  # N*1

            T = K.relu( K.sign(K.dot(one_vector, K.transpose(T2)) - K.dot(T2, K.transpose(one_vector))))  # (Ti(t)>Tj(t)=1); N*N
            T = I2 * T  # only remains T_{ij}=1 when event occured for subject i 1*N

            eta = T * K.exp(-R / 0.1)
            eta = T * K.cast(eta >= T, dtype=tf.float32)
            loss = 1 - K.sum(eta) / (1 + K.sum(T))

            return loss  # + brier_score

        propensity = y_pred[:, 2, :]
        y_pred1 = y_pred[:, 0, :]
        y_pred0 = y_pred[:, 1, :]

        propensity_1 = 1.0 - K.clip(propensity, 0.0001, 1.0)

        partial_loss = partial_loss(y_true[:, 0:window_size * 2],
                                    y_true[:, window_size * 2:window_size * 4], y_pred1, y_pred0)

        rank_loss = rank_loss(y_true[:, 0:window_size * 2],
                              y_true[:, window_size * 2:window_size * 4], y_pred1, y_pred0)

        causal_loss3 = debias_loss(y_true[:, 0:window_size * 2], y_true[:, window_size * 2:window_size * 4], y_pred1,
                                   y_pred0, propensity_1)

        return alpha * partial_loss + beta * rank_loss + causal_loss3 * gamma

    return loss



def prop_likelihood_lrnn(window_size):
    def loss(y_true, y_pred):
        propensity = y_pred,
        propensity_loss =  K.mean((tf.repeat(K.reshape(y_true[:, -1], (-1,1)) , window_size, axis=1) - propensity)**2)
        return propensity_loss
    return loss

def surv_likelihood_lrnn_2(window_size, alpha, beta):
    def loss(y_true, y_pred):
        # Partial likelihood loss
        cens_uncens = 1. + y_true[:, 0:window_size] * (y_pred - 1.)  # component for all patients
        uncens = 1. - y_true[:, window_size:2 * window_size] * y_pred  # component for only uncensored patients
        loss_1 = K.sum(-K.log(K.clip(K.concatenate((cens_uncens, uncens)), K.epsilon(), None)),
                       axis=-1)  # return -log likelihood

        # Rank loss
        R = K.dot(y_pred, K.transpose(y_true[:, 0:window_size]))  # N*N
        diag_R = K.reshape(tf.linalg.diag_part(R), (-1, 1))  # N*1
        one_vector = tf.ones(K.shape(diag_R))  # N*1
        R = K.dot(one_vector, K.transpose(diag_R)) - R  # r_{i}(T_{j}) - r_{j}(T_{j})
        R = K.transpose(R)  # r_{i}(T_{i}) - r_{j}(T_{i})

        I2 = K.reshape(K.sum(y_true[:, window_size:2 * window_size], axis=1), (-1, 1))  # N*1
        I2 = K.cast(K.equal(I2, 1), dtype=tf.float32)
        I2 = K.dot(one_vector, K.reshape(I2,(1,-1)))
        T2 = K.reshape(K.sum(y_true[:, 0:window_size], axis=1), (-1, 1))  # N*1

        T = K.relu(
            K.sign(K.dot(one_vector, K.transpose(T2)) - K.dot(T2, K.transpose(one_vector))))  # (Ti(t)>Tj(t)=1); N*N
        T = I2*T  # only remains T_{ij}=1 when event occured for subject i 1*N

        eta = T * K.exp(-R/0.1)
        eta = T * K.cast(eta>=T, dtype=tf.float32)
        loss_2 = 1-K.sum(eta)/(1+K.sum(T))

        return alpha * loss_1 + beta * loss_2

    return loss


