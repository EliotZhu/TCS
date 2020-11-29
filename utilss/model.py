import os
import pickle

import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow_probability as tfp
tfd = tfp.distributions
tfpl = tfp.layers
tfkl = tf.keras.layers
tfkr = tf.keras.regularizers

from tensorflow_probability.python.distributions import independent as independent_lib
from tensorflow_probability.python.distributions import normal as normal_lib

from keras.callbacks import EarlyStopping
from lifelines import KaplanMeierFitter
from utilss.layers import ExternalMasking, concateDim, reducedim
from utilss.loss_function import surv_likelihood_lrnn, prop_likelihood_lrnn, surv_likelihood_lrnn_2
from utilss.data_handler import DataGenerator, DataGenerator_p

from tqdm.keras import TqdmCallback
tf.keras.backend.set_floatx('float32')


def custom_prior_fn(dtype, shape, name, trainable,
                                   add_variable_fn):
  """Creates multivariate standard `Normal` distribution.
  """
  del name, trainable, add_variable_fn   # unused
  dist = normal_lib.Normal(loc=tf.ones(shape, dtype)*0.01, scale=dtype.as_numpy_dtype(0.1))
  batch_ndims = tf.size(dist.batch_shape_tensor())
  return independent_lib.Independent(dist, reinterpreted_batch_ndims=batch_ndims)



def create_model(input_dim, max_time, history_itvl, data, val_data, lstm_window = 3, alpha=2, beta=2, gamma=1,
                 load = True, verbose = 0, model_name ='dSurv_history.pkl', batch_size = 256, layers = 10):

    train_gen = DataGenerator_p(data, batch_size= int(batch_size/4))
    val_gen = DataGenerator_p(val_data, batch_size= int(batch_size/4))

    input_x = tfkl.Input(shape=(history_itvl, input_dim-1))
    input_m = tfkl.Input(shape=(history_itvl, input_dim-1))

    propensity_layer = ExternalMasking(mask_value=-1)([input_x, input_m])
    propensity_layer = tfkl.LSTM(7, return_sequences=True)(propensity_layer)
    propensity_layer = tfkl.TimeDistributed(tfkl.Dense(max_time))(propensity_layer)
    for i in range(3):
        propensity_layer = tfkl.Dense(max_time)(propensity_layer)
    propensity_layer = concateDim(max_time, name = 'propensity_layer')(propensity_layer)
    model_p = tf.keras.Model(inputs=[input_x, input_m], outputs=propensity_layer)
    model_p.compile(loss= prop_likelihood_lrnn(window_size = max_time), optimizer=tf.keras.optimizers.RMSprop(lr=0.01))
    early_stopping = EarlyStopping(monitor='loss', patience=2)
    model_p.fit(train_gen, validation_data=val_gen, epochs=100, callbacks=[early_stopping], verbose=0)


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
    #combined_layer = ExternalMasking(mask_value=-1)([input_x, input_m])
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
                  optimizer=tf.keras.optimizers.RMSprop(lr=0.001))


    checkpoint_path = os.path.join(os.getcwd(), 'saved_models', 'check_point')


    if load:
        model.load_weights(checkpoint_path)
        with open(os.path.join(os.getcwd(), 'saved_models', model_name), 'rb') as file:
             history_dict = pickle.load(file)
    else:
        cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath = checkpoint_path, save_weights_only=True, verbose=0)
        early_stopping = EarlyStopping(monitor='loss', patience=2)
        history = model.fit(train_gen,validation_data=val_gen,epochs=5,
                            callbacks=[early_stopping,cp_callback, TqdmCallback(verbose=verbose)], verbose=0)

        history_dict = history.history
        with open(os.path.join(os.getcwd(), 'saved_models', model_name), 'wb') as file:
           pickle.dump(history_dict, file)

    return model,model_p,history_dict



def get_counterfactuals(model, data, t = 0, draw = 30, type = "DSurv", test_data = None):
    if type == "DSurv":

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


            y_pred1_t = y_pred_t[:,0,:].copy()
            y_pred1_std = y_pred_std[:,0,:].copy()
            y_pred0_t = y_pred_t[:,1,:].copy()
            y_pred0_std = y_pred_std[:,1,:].copy()

            y_pred_t = y_pred0_t.copy()
            y_pred_t[rnn_x[time_pt == t, 0, 0] == 1] = y_pred1_t[rnn_x[time_pt == t, 0, 0] == 1]
            y_pred_std = y_pred0_std.copy()
            y_pred_std[rnn_x[time_pt == t, 0, 0] == 1] = y_pred1_std[rnn_x[time_pt == t, 0, 0] == 1]

            cf_std_1 =  y_pred1_std.copy()+ y_pred0_std.copy()

            return  y_pred_t, y_pred_std, y_pred1_t, y_pred0_t, cf_std_1, time_pt

        y_pred_t, y_pred_std, y_pred1_t, y_pred0_t, cf_std_1, time_pt = get(data)


        if test_data != None:
            y_pred_t_test, y_pred_std_test, y_pred1_t_test, y_pred0_t_test, cf_std_1_test, time_pt_test = get(test_data)
        else:
            y_pred_t_test, y_pred_std_test, y_pred1_t_test, y_pred0_t_test, cf_std_1_test, time_pt_test = [None,None,None,None,None,None]

    elif  type == "StandardRNN":

        def get(data):
            rnn_x, rnn_m, rnn_s, rnn_y, time_pt = data
            rnn_x0 = rnn_x.copy()
            rnn_x0[:, :, 0][rnn_m[:, :, 0] > 0] = 0
            rnn_x1 = rnn_x.copy()
            rnn_x1[:, :, 0][rnn_m[:, :, 0] > 0] = 1

            y_pred_t = model.predict([rnn_x[time_pt == t], rnn_m[time_pt == t]], verbose=0)
            y_pred0_t = model.predict([rnn_x0[time_pt == t], rnn_m[time_pt == t]], verbose=0)
            y_pred1_t = model.predict([rnn_x1[time_pt == t], rnn_m[time_pt == t]], verbose=0)

            # ATT = y_pred_t[rnn_x[time_pt == t][:, 0, 0] == 1]
            # ATC = y_pred_t[rnn_x[time_pt == t][:, 0, 0] == 0]
            y_pred_std = 0
            cf_std_1 = 0

            return y_pred_t, y_pred_std, y_pred1_t, y_pred0_t, cf_std_1, time_pt
        y_pred_t, y_pred_std, y_pred1_t, y_pred0_t, cf_std_1, time_pt = get(data)

        if test_data != None:
            y_pred_t_test, y_pred_std_test, y_pred1_t_test, y_pred0_t_test, cf_std_1_test, time_pt_test = get(test_data)
        else:
            y_pred_t_test, y_pred_std_test, y_pred1_t_test, y_pred0_t_test, cf_std_1_test, time_pt_test = [None, None,
                                                                                                           None, None,
                                                                                                           None, None]
    elif type == "KM":
        rnn_x, rnn_m, rnn_s, rnn_y, time_pt = data
        y_pred_t, y_pred1_t, y_pred0_t = model
        y_pred_std = 0
        cf_std_1 = 0
        y_pred_t_test, y_pred_std_test, y_pred1_t_test, y_pred0_t_test, cf_std_1_test, time_pt_test = [None, None, None,
                                                                                                       None, None, None]


    return y_pred_t, y_pred_std, y_pred1_t, y_pred0_t, cf_std_1, time_pt,\
           y_pred_t_test, y_pred_std_test, y_pred1_t_test, y_pred0_t_test, cf_std_1_test, time_pt_test



def benchmark_algorithms(input_dim, max_time, history_itvl, data, val_data, one_X = None, lstm_window = 30, model = "StandardRNN", beta = 0,
                         batch_size = 2056):

    rnn_x, rnn_m, rnn_s, rnn_y, _ = data
    rnn_x_val, rnn_m_val, rnn_s_val, rnn_y_val, _ = val_data


    if model == "StandardRNN":
        input_x = tfkl.Input(shape=(history_itvl, input_dim))
        input_m = tfkl.Input(shape=(history_itvl, input_dim))
        #x = ExternalMasking(mask_value=-1)([input_x, input_m])
        x = tfkl.LSTM(lstm_window, return_sequences=True)(input_x)
        for i in range(10):
            x = tfkl.Dense(max_time)(x)
        out = concateDim(max_time)(x)
        model = tf.keras.Model(inputs=[input_x, input_m], outputs=out)
        #model.summary()

        model.compile(loss=surv_likelihood_lrnn_2(max_time, alpha=1, beta=beta),
                      optimizer=tf.keras.optimizers.RMSprop(lr=0.001))

        early_stopping = EarlyStopping(monitor='loss', patience=2)
        model.fit([rnn_x, rnn_m], rnn_y,
                  validation_data=([rnn_x_val, rnn_m_val], rnn_y_val),
                  batch_size=batch_size, epochs=500, callbacks=[early_stopping], verbose=1, shuffle=True)



    elif model == "binaryRNN":

        input_x = tfkl.Input(shape=(history_itvl, input_dim))
        input_m = tfkl.Input(shape=(history_itvl, input_dim))
        #x = ExternalMasking(mask_value=-1)([input_x, input_m])
        x = tfkl.LSTM(lstm_window, return_sequences=True)(input_x)
        for i in range(10):
            x = tfkl.Dense(max_time)(x)
        out = concateDim(max_time)(x)
        model = tf.keras.Model(inputs=[input_x, input_m], outputs=out)

        model.compile(loss= 'mse', optimizer=tf.keras.optimizers.RMSprop(lr=0.001))

        early_stopping = EarlyStopping(monitor='loss', patience=2)
        model.fit([rnn_x, rnn_m], rnn_y[:,0:max_time],
                  validation_data=([rnn_x_val, rnn_m_val], rnn_y_val[:,0:max_time]),
                  batch_size=batch_size, epochs=500, callbacks=[early_stopping], verbose=1, shuffle=True)

    elif model == "KM":
        kmf0 = KaplanMeierFitter().fit(one_X[one_X[:, 3] == 0, 1], event_observed=one_X[one_X[:, 3] == 0, 2])
        kmf1 = KaplanMeierFitter().fit(one_X[one_X[:, 3] == 1, 1], event_observed=one_X[one_X[:, 3] == 1, 2])
        kmf =  KaplanMeierFitter().fit(one_X[:, 1], event_observed=one_X[:, 2])

        model = [kmf, kmf1, kmf0]


    return  model



