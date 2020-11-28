# model.py
#
# Author: Jie Zhu
# Tested with Python version 3.8 and TensorFlow 2.0


import os
import pickle

import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
tfd = tfp.distributions
tfpl = tfp.layers
tfkl = tf.keras.layers
tfkr = tf.keras.regularizers

from tensorflow_probability.python.distributions import independent as independent_lib
from tensorflow_probability.python.distributions import normal as normal_lib
from keras.callbacks import EarlyStopping
from tqdm.keras import TqdmCallback

from utilss.layers import ExternalMasking, concateDim
from utilss.loss_function import surv_likelihood_lrnn, prop_likelihood_lrnn
from utilss.data_handler import DataGenerator, DataGenerator_p



def custom_prior_fn(dtype, shape, name, trainable, add_variable_fn):
  """
  Creates multivariate standard `Normal` distribution.
  """
  del name, trainable, add_variable_fn   # unused
  dist = normal_lib.Normal(loc=tf.ones(shape, dtype)*0.01, scale=dtype.as_numpy_dtype(0.1))
  batch_ndims = tf.size(dist.batch_shape_tensor())
  return independent_lib.Independent(dist, reinterpreted_batch_ndims=batch_ndims)


def create_model(input_dim, max_time, history_itvl, data, val_data, lstm_window = 3, alpha=2, beta=2, gamma=1,
                 load = False, verbose = 0, model_name ='dSurv_history.pkl', batch_size = 256, layers = 10):
    """
     Define the main network and propensity network
    """
    train_gen = DataGenerator_p(data, batch_size= int(batch_size/4))
    val_gen = DataGenerator_p(val_data, batch_size= int(batch_size/4))

    print('training propensity network...')

    input_x = tfkl.Input(shape=(history_itvl, input_dim))
    input_m = tfkl.Input(shape=(history_itvl, input_dim))

    propensity_layer = ExternalMasking(mask_value=-1)([input_x, input_m])
    propensity_layer = tfkl.LSTM(7, return_sequences=True)(propensity_layer)
    propensity_layer = tfkl.TimeDistributed(tfkl.Dense(max_time))(propensity_layer)
    for i in range(3):
        propensity_layer = tfkl.Dense(max_time)(propensity_layer)
    propensity_layer = concateDim(max_time, name = 'propensity_layer')(propensity_layer)
    model_p = tf.keras.Model(inputs=[input_x, input_m], outputs=propensity_layer)
    model_p.compile(loss= prop_likelihood_lrnn(window_size = max_time), optimizer=tf.keras.optimizers.RMSprop(lr=0.01))
    early_stopping = EarlyStopping(monitor='loss', patience=2)
    checkpoint_path = os.path.join(os.getcwd(), 'saved_models', 'check_point2')
    cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path, save_weights_only=True, verbose=0)
    model_p.fit(train_gen, validation_data=val_gen, epochs=100,
                callbacks=[early_stopping,cp_callback, TqdmCallback(verbose=verbose)], verbose=0)

    print('training potential outcome network...')

    train_gen = DataGenerator(data, batch_size=batch_size)
    val_gen = DataGenerator(val_data, batch_size=batch_size)

    input_x = tfkl.Input(shape=(history_itvl, input_dim))
    input_m = tfkl.Input(shape=(history_itvl, input_dim))
    input_m0 = tfkl.Input(shape=(history_itvl, input_dim))
    input_m1 = tfkl.Input(shape=(history_itvl, input_dim))

    input_s = tfkl.Input(shape=(history_itvl, input_dim))
    input_x1 = tfkl.Input(shape=(history_itvl, input_dim))
    input_x0 = tfkl.Input(shape=(history_itvl, input_dim))

    control_layer = ExternalMasking(mask_value=-1)([input_x, input_m1])
    control_layer = tfkl.LSTM(lstm_window, return_sequences=True)(control_layer)
    control_layer = tfkl.TimeDistributed(tfkl.Dense(max_time))(control_layer)
    for i in range(layers):
        control_layer = tfkl.Dense(max_time)(control_layer)


    treat_layer = ExternalMasking(mask_value=-1)([input_x, input_m0])
    treat_layer = tfkl.LSTM(lstm_window, return_sequences=True)(treat_layer)
    treat_layer = tfkl.TimeDistributed(tfkl.Dense(max_time))(treat_layer)
    for i in range(layers):
        treat_layer = tfkl.Dense(max_time)(treat_layer)

    combined_layer = tfkl.concatenate([control_layer, treat_layer])
    propensity_layer = model_p([input_x, input_m], training=False)
    combined_layer = tfkl.LSTM(lstm_window, return_sequences=True)(combined_layer)
    combined_layer = tfkl.TimeDistributed(tfkl.Dense(max_time))(combined_layer)
    combined_layer = tfpl.DenseFlipout(max_time,kernel_prior_fn = custom_prior_fn,kernel_posterior_fn =  custom_prior_fn)(combined_layer)
    for i in range(layers):
        combined_layer = tfkl.Dense(max_time)(combined_layer)

    c_x1 = ExternalMasking(mask_value=-1)([input_x1, input_m0])
    c_x1 = tfkl.LSTM(lstm_window, return_sequences=True)(c_x1)
    c_x1 = tfkl.TimeDistributed(tfkl.Dense(max_time))(c_x1)
    c_x1 = tfkl.concatenate([combined_layer, c_x1])
    for i in range(layers):
        c_x1 = tfkl.Dense(max_time)(c_x1)
    c_x1 = concateDim(max_time, name = 'c1')(c_x1)


    c_x0 = ExternalMasking(mask_value=-1)([input_x0, input_m1])
    c_x0 = tfkl.LSTM(lstm_window, return_sequences=True)(c_x0)
    c_x0 = tfkl.TimeDistributed(tfkl.Dense(max_time))(c_x0)
    c_x0 = tfkl.concatenate([combined_layer, c_x0])
    for i in range(layers):
        c_x0 = tfkl.Dense(max_time)(c_x0)
    c_x0 = concateDim(max_time, name = 'c0')(c_x0)

    combined = tf.stack([c_x1, c_x0, propensity_layer], axis=1)
    model = tf.keras.Model(inputs=[input_x, input_m, input_s, input_x0, input_x1, input_m0, input_m1], outputs=combined)
    model.compile(loss=surv_likelihood_lrnn(max_time, alpha=alpha, beta=beta, gamma=gamma),
                  optimizer=tf.keras.optimizers.RMSprop(lr=0.0005))


    checkpoint_path = os.path.join(os.getcwd(), 'saved_models', 'check_point')


    if load:
        model.load_weights(checkpoint_path)
        with open(os.path.join(os.getcwd(), 'saved_models', model_name), 'rb') as file:
             history_dict = pickle.load(file)
    else:
        cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath = checkpoint_path, save_weights_only=True, verbose=0)
        early_stopping = EarlyStopping(monitor='loss', patience=2)
        history = model.fit(train_gen,validation_data=val_gen,epochs=100,
                            callbacks=[early_stopping,cp_callback, TqdmCallback(verbose=verbose)], verbose=0)

        history_dict = history.history
        with open(os.path.join(os.getcwd(), 'saved_models', model_name), 'wb') as file:
           pickle.dump(history_dict, file)

    return model, history_dict



def get_counterfactuals(model, data, t = 0, draw = 30, test_data = None):
    """
        Compute counterfactuals
    """
    def get(data):
        rnn_x, rnn_m, rnn_s, rnn_y, time_pt = data

        rnn_x0 = rnn_x.copy()
        rnn_x0[:, :, 0] = 0
        rnn_x1 = rnn_x.copy()
        rnn_x1[:, :, 0] = 1

        rnn_m0 = rnn_m.copy()
        rnn_m1 = rnn_m.copy()
        rnn_m0[rnn_x[:, :, 0] == 0] = -1
        rnn_m1[rnn_x[:, :, 0] == 1] = -1

        y_pred =  [model.predict([rnn_x[time_pt == t], rnn_m[time_pt == t], rnn_s[time_pt == t],
                                  rnn_x0[time_pt == t], rnn_x1[time_pt == t], rnn_m0[time_pt == t],rnn_m1[time_pt == t] ], verbose=0) for _ in range(draw)]

        y_pred = np.array(y_pred)
        y_pred_t = np.mean(y_pred,0)
        y_pred_std = np.std(y_pred,0)

        # temp = y_pred_t[:, 2, :]
        #
        # plt.hist(rnn_x[:,:,0].reshape(-1,1))
        # plt.hist(temp[:,-1].reshape(-1,1))
        # plt.show()

        #rnn_x[:, :, 0]
        #y = np.concatenate([rnn_y, rnn_x[:, 0:1, 0]], axis=1)
        #temp = np.repeat(np.reshape(y[:, -1], (-1, 1)), 30, axis=1)

        y_pred1_t = y_pred_t[:,0,:].copy()
        y_pred1_std = y_pred_std[:,0,:].copy()
        y_pred0_t = y_pred_t[:,1,:].copy()
        y_pred0_std = y_pred_std[:,1,:].copy()

        y_pred_t = y_pred0_t.copy()
        y_pred_t[rnn_x[time_pt == t, 0, 0] == 1] = y_pred1_t[rnn_x[time_pt == t, 0, 0] == 1]
        y_pred_std = y_pred0_std.copy()
        y_pred_std[rnn_x[time_pt == t, 0, 0] == 1] = y_pred1_std[rnn_x[time_pt == t, 0, 0] == 1]

        cf_std_1 =  y_pred1_std.copy()+ y_pred0_std.copy() #np.std(y_pred1,0) + np.std(y_pred0,0)

        return  y_pred_t, y_pred_std, y_pred1_t, y_pred0_t, cf_std_1, time_pt

    y_pred_t, y_pred_std, y_pred1_t, y_pred0_t, cf_std_1, time_pt = get(data)


    if test_data != None:
        y_pred_t_test, y_pred_std_test, y_pred1_t_test, y_pred0_t_test, cf_std_1_test, time_pt_test = get(test_data)
    else:
        y_pred_t_test, y_pred_std_test, y_pred1_t_test, y_pred0_t_test, cf_std_1_test, time_pt_test = [None,None,None,None,None,None]


    return y_pred_t, y_pred_std, y_pred1_t, y_pred0_t, cf_std_1, time_pt,\
           y_pred_t_test, y_pred_std_test, y_pred1_t_test, y_pred0_t_test, cf_std_1_test, time_pt_test



