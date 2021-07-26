# data_handler.py
#
# Author: Elliott Jie Zhu

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from joblib import Parallel, delayed
from tqdm import tqdm
from scipy.stats import expon, norm


def build_data_rnn(df_x, history_itvl=8, prediction_itvl=1, max_time=0):
    observation = df_x.ID
    one_X = df_x.groupby(df_x.ID)['T'].max()
    event = df_x.groupby(df_x.ID)['Y'].max()
    df_x.drop(['T', 'Y', 'ID'], axis=1, inplace=True)
    train_x = np.array(df_x.reset_index(drop=True))
    x_dim = train_x.shape[1]

    f = event == 1

    def the_loop(i):
        rnn_y = np.empty((0, max_time * 2), dtype=np.float32)
        rnn_x = np.empty((0, history_itvl, x_dim), dtype=np.float32)
        Time = np.empty((0), dtype=np.float32)

        max_engine_time = int(one_X[i])
        yobs = np.zeros((2, max_time), dtype=np.float32)

        if f[i]:
            yobs[0, 0:max_engine_time] = 1.0
            try:
                yobs[1, max_engine_time] = 1.0
            except:
                pass
        else:
            yobs[0, 0:max_engine_time + 1] = 1.0

        start = max(0, max_engine_time - prediction_itvl)
        end = max(max_engine_time, 1)
        step = 1
        count = 0
        for j in np.arange(end, start, -step):
            ytemp = np.zeros((1, max_time * 2))
            y_end = min(j + 1, max_engine_time + 1)
            yobs_temp = yobs.copy()
            yobs_temp[0, y_end:] = 0
            yobs_temp[1, y_end:] = 0
            ytemp[0, 0:max_time] = yobs_temp[0, 0:max_time]
            ytemp[0, max_time:] = yobs_temp[1, 0:max_time]

            if sum(ytemp[0, prediction_itvl:prediction_itvl * 2]) > 0:
                _event = np.array(1).reshape(1)
            else:
                _event = np.array(0).reshape(1)

            rnn_y = np.concatenate((rnn_y, ytemp))
            Time = np.concatenate((Time, np.array(count).reshape(1)))
            count = count + 1

        # step = 1
        # count = 0
        # for j in np.arange(history_itvl, 0, -step):
        covariate_x = train_x[observation == i, :]
        xtemp = np.ones((1, history_itvl, x_dim), dtype=np.float32) * -1
        x_end = history_itvl  # min(j + 1, max_engine_time)
        x_start = 0  # max(x_end - history_itvl, 0)
        x_end_t = covariate_x[x_start:x_end, :].reshape((1, -1, x_dim)).shape
        xtemp[0, 0:x_end_t[1], :] = covariate_x[x_start:x_end, :].reshape((1, -1, x_dim))

        if sum(ytemp[0, prediction_itvl:prediction_itvl * 2]) > 0:
            _event = np.array(1).reshape(1)
        else:
            _event = np.array(0).reshape(1)
        rnn_x = np.concatenate((rnn_x, xtemp))
        # count = count + 1

        return rnn_y, rnn_x, Time

    the_list = Parallel(n_jobs=-1)(delayed(the_loop)(i) for i in tqdm(set(observation)))

    rnn_y = np.concatenate([item[0] for item in the_list])
    rnn_x = np.concatenate([item[1] for item in the_list])
    Time = np.concatenate([item[2] for item in the_list])
    rnn_x[np.isnan(rnn_x)] = -1
    rnn_y[np.isnan(rnn_y)] = 1

    rdata = rnn_x, rnn_y
    return rdata, Time


def LDataSimu(seed=1234, sampleSize=500, max_time=10, history_itvl=5,
              simu_dim=4, scale=1, confound=1, std=0.5, dynamic=False):
    np.random.seed(seed)
    scale = np.int(scale)

    def outcome_func_gen(model="1"):
        if model == "1":
            def outcome_func(gps, A, noise=True):
                if noise:
                    Y = A + gps + norm.rvs(size=len(gps), scale=0.05)
                else:
                    Y = A + gps
                return Y
        elif model == "2":
            def outcome_func(gps, A, noise=True):
                if noise:
                    Y = A + (gps) + norm.rvs(size=len(gps), scale=0.1)
                else:
                    Y = A + (gps)
                return Y / 10
        else:
            print('Model not found')
            outcome_func = None
        return outcome_func

    def gps_func_gen(model="1"):
        if model == "1":
            def gps_func(X, A):
                return expon(scale=1 / np.sum(X, 1)).pdf(A)
        elif model == "2":
            def gps_func(X, A):
                return np.sum(X, 1)
        else:
            print('Model not found')
            gps_func = None
        return gps_func

    def trt_func_gen(model="1"):
        if model == "1":
            def trt_func(X):
                P = np.sum(X[:, :3], 1) > np.sum(X[:, 3:-1], 1)
                P = np.clip(confound * P + (1 - confound) * 0.5, a_min=0, a_max=1)
                A = np.array([np.random.choice([1, 0], 1, p=[1 - p, p]) for p in P]).reshape(-1)

                return A

        elif model == "2":
            def trt_func(X):
                return norm.rvs(loc=1 / np.sum(X, 1), scale=0) / (simu_dim)
        else:
            print('Model not found')
            trt_func = None
        return trt_func

    def X_generator_gen(model="1"):
        # Please ensure X and A are normlaised above 1.
        if model == "1":
            def X_generator(N, t):
                return expon.rvs(loc=10, scale=2, size=N)
        elif model == "2":
            def X_generator(N, t):
                return norm.rvs(size=N, scale=std, loc=t)
        else:
            print('Model not found')
            X_generator = None
        return X_generator

    gps_func = gps_func_gen("2")
    X_generator = X_generator_gen("2")
    trt_func = trt_func_gen("1")
    outcome_func = outcome_func_gen("2")

    def x_seed(simu_dim, sampleSize, t):
        X = []
        for j in range(simu_dim):
            xtemp = X_generator(sampleSize, t) / simu_dim
            X.append(xtemp.reshape(-1, 1))
        X = np.concatenate(X, 1)
        return X

    def traj(t, at=None):
        t_effect = (t) ** (1 / scale)
        Xt = x_seed(simu_dim, sampleSize, t_effect)
        if at is None:
            at = trt_func(Xt.copy())
        gps = gps_func(Xt, at)
        return at, Xt, gps

    x_series = []
    a_series = []
    p_series = []
    for t in range(0, history_itvl):
        if t == 0:
            at, xt, pt = traj(t)
        else:
            # xt = x_t(t, this_x.copy())
            # xt = this_x.copy()
            if dynamic:
                at, xt, pt = traj(t)
            else:
                at, xt, pt = traj(t, at=at)
            # print(np.mean(np.sum(xt,1)))
            # print(np.mean(yt))
        a_series.append(at.reshape(sampleSize, 1))
        p_series.append(pt.reshape(sampleSize, 1))
        x_series.append(xt.reshape(sampleSize, 1, simu_dim))

    x_series = np.concatenate(x_series, 1)
    a_series = np.concatenate(a_series, 1)
    p_series = np.concatenate(p_series, 1)

    # y_series = np.mean(y_series[:,0,:])/y_series
    p_series = p_series.reshape(sampleSize, history_itvl, 1)
    a_series = a_series.reshape(sampleSize, history_itvl, 1)

    train_x = np.append(a_series, x_series, 2)
    train_x = np.append(train_x, p_series, 2)

    # import statsmodels.api as sm
    # gamma_model = sm.GLM(a_series.reshape(-1), x_series.reshape(-1,simu_dim), family=sm.families.family.Binomial())
    # gamma_results = gamma_model.fit()
    # est_treat = gamma_results.predict(x_series)
    # #plt.hist(est_treat[a_series.reshape(-1)==1],alpha=0.5,bins=50)
    # plt.hist(est_treat.reshape(-1),alpha=0.5,bins=50)
    # plt.show()
    #

    y_series = []
    for t in range(0, max_time):
        t_effect = (t) ** (1 / scale)
        yt = outcome_func(np.mean(p_series, 1) * t_effect, np.mean(a_series, 1), noise=False)
        y_series.append(yt.reshape(sampleSize, 1))
    y_series = np.concatenate(y_series, 1)
    y_series = y_series.reshape(sampleSize, max_time, 1)

    def surv_func_wrapper(y_series, mode='constant'):
        # hazard
        if mode == 'constant':
            hazard = np.repeat(np.zeros((1, max_time)).reshape(1, -1), sampleSize, 0) + y_series[:, :, 0]
        else:
            hazard = (y_series[:, :, 0]) + np.repeat(np.arange(0, max_time).reshape(1, -1), sampleSize, 0) * 0.001

        hazard[:, 0] = 0
        trueSurv = np.clip(np.exp(- hazard), a_min=0, a_max=1)
        return trueSurv

    trueSurv = surv_func_wrapper(y_series, mode='')

    # print('Simulate failure time')
    trueT = np.ones((sampleSize)) * max_time
    trueC = np.ones((sampleSize)) * max_time
    event = np.zeros((sampleSize))
    censor = np.zeros((sampleSize))

    for t in range(0, max_time):
        # T = np.array([integrate.quad(surv_func_t, t-1, t, args=(x_i, a_i))[0] for x_i, a_i in zip(seed_x, A)])
        T = trueSurv[:, t] < np.random.uniform(0, 1, sampleSize)
        trueT[(T == True) & (event == 0)] = t
        event[T == True] = 1
        C = np.random.uniform(0, 1, sampleSize) < np.random.uniform(0, 1, sampleSize) * 0.01
        trueC[(C == True) & (censor == 0)] = t
        censor[C == True] = 1

    time = np.minimum(trueT, trueC)
    event = (time == trueT) & (event == 1)
    time = np.round(time, 0)

    # Want DF?
    train_df = pd.DataFrame(np.concatenate(train_x, 0))
    train_df.columns = np.arange(0, len(train_df.columns)).astype(str)
    train_df.rename(columns={"0": "A"}, inplace=True)
    train_df.rename(columns={train_df.columns[-1]: "gps"}, inplace=True)
    train_df['T1'] = np.repeat(np.arange(0, history_itvl, 1).reshape(-1, 1), sampleSize, 1).T.reshape(-1)
    train_df['ID'] = np.repeat(np.arange(0, sampleSize, 1), history_itvl)
    # train_df['Delta'] = event * 1.0
    full_df = train_df.copy()
    full_df['Y'] = np.repeat(event, history_itvl)
    full_df['T'] = np.repeat(time, history_itvl)

    # X = train_df.reset_index(drop=True)
    # idx_censor = []
    # for id in (np.unique(X['ID'])):
    #     temp = X.loc[X['ID'] == id]['T']
    #     temp_idx = temp.index <= temp[temp <= time[int(id)]].index[-1]
    #     temp_idx = temp.index[temp_idx]
    #     temp_idx = np.array(temp_idx)
    #     idx_censor.append(temp_idx)
    # idx_censor = np.concatenate(idx_censor)
    # X = X.iloc[idx_censor].reset_index(drop=True)
    # T = pd.DataFrame(np.array(X['T']), columns=['T']).reset_index(drop=True)
    # df = X.copy()
    # df['Y'] = 0
    # df['C'] = 0
    # for i in (np.unique(df['ID'])):
    #     temp = np.zeros(len(df.loc[df['ID'] == i, 'Y']))
    #     temp[-1] = event[int(i)] * 1
    #     df.loc[df['ID'] == i, 'Y'] = temp
    #     temp = np.zeros(len(df.loc[df['ID'] == i, 'C']))
    #     temp[-1] = 1 - event[int(i)] * 1
    #     df.loc[df['ID'] == i, 'C'] = temp

    df = full_df.drop(['gps'], axis=1)

    return df, trueSurv


def dataset_normalize(_dataset, all_x_add):
    def get_mean(x):
        x_mean = []
        for i in range(x.shape[0]):
            mean = np.mean(x[i])
            x_mean.append(mean)
        return x_mean

    def get_std(x):
        x_std = []
        for i in range(x.shape[0]):
            std = np.std(x[i])
            x_std.append(std)
        return x_std

    train_proportion = 0.8
    train_index = int(all_x_add.shape[1] * train_proportion)
    train_x = all_x_add[:, :train_index]
    x_mean = get_mean(train_x)
    x_std = get_std(train_x)

    x_mean = np.asarray(x_mean)
    x_std = np.asarray(x_std)

    for i in tqdm(range(_dataset.shape[0])):
        _dataset[i][0] = (_dataset[i][0] - x_mean[:, None])
        _dataset[i][0] = _dataset[i][0] / x_std[:, None]

    return _dataset


class DataGenerator(tf.keras.utils.Sequence):
    def __init__(self, data, batch_size=32, shuffle=True):
        self.rnn_x, self.rnn_y = data
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.on_epoch_end()

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(len(self.rnn_x) / self.batch_size))

    def __getitem__(self, index):
        'Generate one batch of data'
        indexes = self.indexes[index * self.batch_size:(index + 1) * self.batch_size]
        X, y = self.__data_generation(indexes)
        return X, y

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(len(self.rnn_x))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def __data_generation(self, indexes):
        'Generates data containing batch_size samples'

        return self.rnn_x[indexes], self.rnn_y[indexes]


class DataGenerato_PS(tf.keras.utils.Sequence):
    def __init__(self, data, batch_size=32, shuffle=True):
        self.rnn_x = data
        self.batch_size = batch_size
        self.shuffle = shuffle

        self.on_epoch_end()

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(len(self.rnn_x) / self.batch_size))

    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        indexes = self.indexes[index * self.batch_size:(index + 1) * self.batch_size]
        # Generate data
        X, y = self.__data_generation(indexes)

        return X, y

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(len(self.rnn_x))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def __data_generation(self, indexes):
        'Generates data containing batch_size samples'  # X : (n_samples, *dim, n_channels)
        # Initialization
        y = self.rnn_x[indexes, 0].reshape(-1)
        x = self.rnn_x[indexes, 1:]
        return x, y
