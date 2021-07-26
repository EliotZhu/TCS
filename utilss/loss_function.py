# loss_function.py
#
# Author: Elliott Jie Zhu
# Tested with Python version 3.7 and Keras version 2 (using TensorFlow backend)

import tensorflow as tf
from tensorflow.keras import backend as K

def surv_likelihood_lrnn(window_size, alpha, beta):
    @tf.autograph.experimental.do_not_convert
    def loss(y_true, y_pred):
        def partial_loss(y_true, y_pred):

            y_true_p1 = y_true[:, 0:window_size]
            y_true_p2 = y_true[:, window_size:]

            cens_uncens = 1. + (y_pred - 1.) * y_true_p1  # component for all patients
            uncens = 1. - y_pred * y_true_p2  # component for uncensored patients
            loss = K.sum(-K.log(K.clip(K.concatenate((cens_uncens, uncens)), K.epsilon(), None)),
                         -1)  # return -log likelihood
            return loss

        def rank_loss(y_true, y_pred):
            y_true_p1 = y_true[:, 0:window_size]
            y_true_p2 = y_true[:, window_size:]

            y_true_p1 = K.reshape(y_true_p1, [-1, window_size])
            y_true_p2 = K.reshape(y_true_p2, [-1, window_size])
            y_pred = K.reshape(y_pred, [-1, window_size])

            R = tf.matmul(y_true_p2, K.transpose(y_true_p1))

            P = tf.matmul(y_true_p2, K.transpose(y_pred))
            diag_P = tf.linalg.tensor_diag_part(P)
            temp = - K.sign(P - K.reshape(diag_P, [-1, 1])) * R

            loss = K.sum(temp) / K.sum(R * 1.0)

            return loss

        partial_loss = partial_loss(y_true, y_pred)

        rank_loss = rank_loss(y_true, y_pred)


        return alpha * partial_loss + beta * rank_loss

    return loss


