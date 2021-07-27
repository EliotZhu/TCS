import numpy as np
import tensorflow as tf

tfkl = tf.keras.layers
tfkr = tf.keras.regularizers
import kerastuner as kt

from tensorflow.keras.callbacks import EarlyStopping
from utilss.loss_function import surv_likelihood_lrnn
from utilss.data_handler import DataGenerator, DataGenerato_PS, build_data_rnn

tf.keras.backend.set_floatx('float32')


class MyTuner(kt.tuners.BayesianOptimization):

    def __init__(self,
                 train_data,
                 val_data,
                 DataGenerator,
                 sample_size,
                 **kwargs):
        self.train_data = train_data
        self.val_data = val_data
        self.sample_size = sample_size
        self.DataGenerator = DataGenerator

        super(MyTuner, self, ).__init__(**kwargs)

    def run_trial(self, trial, *args, **kwargs):
        # You can add additional HyperParameters for preprocessing and custom training loops
        kwargs['x'] = self.DataGenerator(self.train_data,
                                         batch_size=trial.hyperparameters.Int('batch_size', int(self.sample_size / 10),
                                                                              int(self.sample_size / 2),
                                                                              step=int(self.sample_size / 10)))
        kwargs['validation_data'] = self.DataGenerator(self.val_data,
                                                       batch_size=trial.hyperparameters.Int('batch_size',
                                                                                            int(self.sample_size / 10),
                                                                                            int(self.sample_size / 2),
                                                                                            step=int(self.sample_size / 10)))
        # kwargs['epochs'] = trial.hyperparameters.Int('epochs', 10, 30)
        super(MyTuner, self).run_trial(trial, *args, **kwargs)


# def custom_prior_fn(dtype, shape, name, trainable,
#                     add_variable_fn):
#     """Creates multivariate standard `Normal` distribution.
#     """
#     del name, trainable, add_variable_fn  # unused
#     dist = normal_lib.Normal(loc=tf.ones(shape, dtype) * 0.0, scale=dtype.as_numpy_dtype(0.1))
#     batch_ndims = tf.size(dist.batch_shape_tensor())
#     return independent_lib.Independent(dist, reinterpreted_batch_ndims=batch_ndims)
#

def create_model(simu_dim, history_itvl, max_time):
    def ps_model(hp):

        input_x = tfkl.Input(shape=(simu_dim))
        mask_layer = tfkl.Masking(mask_value=-1)(input_x)
        # l1= tfkl.TimeDistributed(tfkl.Dense(max_time, activation='relu'))(mask_layer)
        # l1 = tfkl.Flatten()(l1)
        l1 = tfkl.Dense(max_time)(input_x)
        for i in range(hp.Int('layers', 3, 10, default=5)):
            l1 = tfkl.Dense(hp.Int('unit', simu_dim, int(simu_dim *3), default=simu_dim))(l1)
        l1 = tfkl.Dense(hp.Int('unit', simu_dim, int(simu_dim * 3), default=simu_dim))(l1)
        ps = tfkl.Dense(1, activation='sigmoid')(l1)
        model = tf.keras.Model(inputs=[input_x], outputs=ps)

        model.compile(loss=tf.keras.losses.binary_crossentropy,
                      optimizer=tf.keras.optimizers.RMSprop(learning_rate=0.0005))
        return model

    def surv_model(hp):

        input_x = tfkl.Input(shape=(history_itvl, 2))
        encode = tfkl.Masking(mask_value=-1)(input_x)
        encode = tfkl.TimeDistributed(tfkl.Dense(history_itvl, activation='relu'))(encode)
        encode = tfkl.Flatten()(encode)
        decode = tfkl.Dense(history_itvl)(encode)
        for i in range(hp.Int('layers', 3, 20, default=5)):
            decode = tfkl.Dense(hp.Int('unit', max_time, max_time * 4, default=max_time))(decode)
        decode = tfkl.Dense(max_time, activation='sigmoid')(decode)
        model = tf.keras.Model(inputs=[input_x], outputs=decode)
        model.compile(loss=surv_likelihood_lrnn(max_time, alpha=1, beta=0.2),
                      optimizer=tf.keras.optimizers.RMSprop(learning_rate=0.001))

        return model

    return ps_model, surv_model


def fit_model(model, max_time, simu_dim, history_itvl, train, val):
    print('search hype...', end = '')
    ps_model = model[0]
    surv_model = model[1]

    tuner1 = MyTuner(
        hypermodel=ps_model,
        objective='loss',
        max_trials=5,
        overwrite=True,
        train_data=np.array(train)[:, 0:simu_dim + 1].astype('float32'),
        val_data=np.array(val)[:, 0:simu_dim + 1].astype('float32'),
        DataGenerator=DataGenerato_PS,
        sample_size=len(train))
    tuner1.search(epochs=5, callbacks=[EarlyStopping(monitor='loss', patience=2)], verbose=0)
    best_hyperparameters_ps = tuner1.get_best_hyperparameters(1)[0]
    best_hyperparameters_ps.values
    train_gen = DataGenerato_PS(np.array(train)[:, 0:simu_dim + 1].astype('float32'),
                                batch_size=best_hyperparameters_ps.values['batch_size'])
    best_ps_models = []
    for i in range(1):
        print('.', end='')
        best_ps_model = tuner1.hypermodel.build(best_hyperparameters_ps)
        early_stopping = EarlyStopping(monitor='loss', patience=2)
        best_ps_model.fit(train_gen, epochs=100, callbacks=[early_stopping], verbose=0)
        best_ps_models.append(best_ps_model)

    def get_surv_ps_data(train, best_ps_models, simu_dim, max_time, history_itvl):
        ps_pred = [model.predict(np.array(train)[:, 1:simu_dim + 1].astype('float32'), verbose=0) for model in (best_ps_models)]
        ps_pred = np.mean(np.array(ps_pred), 0).reshape(-1)
        #ps_pred[train['A'] == 0] =  1-ps_pred[train['A'] == 0]
        surv_train = train[['A', 'ID', 'T', 'Y']].copy()
        surv_train['PS'] = ps_pred

        data, Time = build_data_rnn(surv_train.copy(), history_itvl=history_itvl, prediction_itvl=1, max_time=max_time)

        return data, Time

    data, Time = get_surv_ps_data(train, best_ps_models, simu_dim, max_time, history_itvl)
    val_data, Time_val = get_surv_ps_data(val, best_ps_models, simu_dim, max_time, history_itvl)


    # Surv model
    tuner = MyTuner(
        hypermodel=surv_model,
        objective='loss',
        max_trials=8,
        overwrite=True,
        train_data=data,
        val_data=val_data,
        DataGenerator=DataGenerator,
        sample_size=len(data[0]))

    tuner.search(epochs=5, callbacks=[EarlyStopping(patience=1,monitor='loss')], verbose=0)

    best_hyperparameters_x = tuner.get_best_hyperparameters(1)[0]
    train_gen = DataGenerator(data, batch_size=best_hyperparameters_x.values['batch_size'])
    best_outcome_model = []
    for i in (range(20)):
        print('.', end='')
        best_outcome_model_x = tuner.hypermodel.build(best_hyperparameters_x)
        early_stopping = EarlyStopping(monitor='loss', patience=2)
        best_outcome_model_x.fit(train_gen, epochs=100, callbacks=[early_stopping], verbose=0)
        best_outcome_model.append(best_outcome_model_x)

    print('done!')
    return best_ps_models, best_outcome_model


# models = best_outcome_model
def get_counterfactuals(best_models, df, simu_dim, max_time, history_itvl):
    def get_surv_ps_data(train, best_ps_models):
        ps_pred = [model.predict(np.array(train)[:, 1:simu_dim + 1], verbose=0) for model in (best_ps_models)]
        ps_pred = np.mean(np.array(ps_pred), 0)
        surv_train0 = train[['A', 'ID', 'T', 'Y']].copy()
        surv_train1 = train[['A', 'ID', 'T', 'Y']].copy()

        ps_pred0 = ps_pred.copy()
        ps_pred0[train['A'] == 1] = 1-ps_pred0[train['A'] == 1]
        surv_train0['PS'] = ps_pred0
        surv_train1['PS'] = ps_pred

        data0, Time = build_data_rnn(surv_train0.copy(), history_itvl=history_itvl, prediction_itvl=1,max_time=max_time)
        data1, _ = build_data_rnn(surv_train1.copy(), history_itvl=history_itvl, prediction_itvl=1, max_time=max_time)

        return data1, data0

    def get(df):
        best_ps_models, best_outcome_model = best_models

        pred_data_1,pred_data_0 = get_surv_ps_data(df.astype('float32'), best_ps_models)

        rnn_x0, rnn_y = pred_data_1
        rnn_x1, _ = pred_data_0
        rnn_m = rnn_x0 == -1

        rnn_x0[:, :, 0] = 0
        rnn_x0[rnn_m] = -1
        rnn_x1[:, :, 0] = 1
        rnn_x1[rnn_m] = -1

        # rnn_x1[rnn_x[:, :, 0] == 0] = -1
        # rnn_x0[rnn_x[:, :, 0] == 1] = -1
        print('predicting...', end='')

        y_pred1_t =  np.array([model.predict([rnn_x1], verbose=0) for model in (best_outcome_model)])
        y_pred0_t =  np.array([model.predict([rnn_x0], verbose=0) for model in (best_outcome_model)])


        print(np.mean(np.mean(y_pred1_t, 0),0)-np.mean(np.mean(y_pred0_t, 0),0))


        surv0 = y_pred0_t.copy()
        surv1 = y_pred1_t.copy()


        surv_ind_0 = np.mean(np.cumprod(surv0, 2), 0)
        surv_ind_std_0 = np.std(np.cumprod(surv0, 2), 0)

        surv_ind_1 = np.mean(np.cumprod(surv1, 2), 0)
        surv_ind_std_1 = np.std(np.cumprod(surv1, 2), 0)

        # plt.plot(np.mean(surv_ind_0, 0), color='grey')
        # plt.plot(np.mean(surv_ind_1, 0), color='blue')
        # plt.show()

        ite = np.mean(np.cumprod(y_pred1_t, 2) - np.cumprod(y_pred0_t, 2), 0)
        ite_std = np.std(np.cumprod(y_pred1_t, 2) - np.cumprod(y_pred0_t, 2), 0)

        ate_temp = np.mean(np.cumprod(y_pred1_t, 2) - np.cumprod(y_pred0_t, 2), 1)
        ate = np.mean(ate_temp, 0)
        ate_std = np.std(ate_temp, 0)

        hr_temp = np.mean(np.clip(np.cumprod(y_pred1_t, 2), a_min=0.1, a_max=1)
                          / np.clip(np.cumprod(y_pred0_t, 2), a_min=0.1, a_max=1), 1)
        hr = np.mean(hr_temp, 0)
        hr_std = np.std(hr_temp, 0)
        print('done!')

        return surv_ind_0, surv_ind_1,surv_ind_std_0,surv_ind_std_1, ite, ite_std, ate, ate_std, hr, hr_std

    return get(df)



