# data_handler.py
#
# Author: Jie Zhu
# Tested with Python version 3.8 and TensorFlow 2.0

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from lifelines import KaplanMeierFitter
from tqdm import tqdm
import tensorflow as tf


def arr_to_xmd(arr, fill = False):
    m_temp = np.isnan(arr)
    m_temp = m_temp[0]
    arr = arr[0]

    s_temp = m_temp * 1.0
    c = np.cumsum(s_temp, axis=0)
    d = np.vstack( (np.zeros((1,s_temp.shape[1])), np.where(s_temp == 0, c, 0) ) )
    np.maximum.accumulate(d, axis=0, out=d)
    e = np.diff(d, axis=0) * (1 - s_temp)
    f = (s_temp.copy() - e)
    s_temp = f.cumsum(axis=0)

    if fill:
        idx = np.where(~m_temp.T, np.arange(m_temp.shape[0]), 0).T
        np.maximum.accumulate(idx, axis=0, out=idx)
        out = arr[idx.T, np.arange(idx.shape[1])[:, None]]
        x_temp = out.T
    else:
        x_temp = arr
    m_temp  = 1.0 - m_temp * 1.0
    row = x_temp.shape[0]
    col = x_temp.shape[1]
    return x_temp.reshape((1,row,col)), m_temp.reshape((1,row,col)),s_temp.reshape((1,row,col))


def make_surv_array(t, f, breaks):
    """Transforms censored survival data into vector format that can be used in Keras.
      Arguments
          t: Array of failure/censoring times.
          f: Censoring indicator. 1 if failed, 0 if censored.
          breaks: Locations of breaks between time intervals for discrete-time survival model (always includes 0)
      Returns
          Two-dimensional array of survival data, dimensions are number of individuals X number of time intervals*2
    """
    n_samples = t.shape[0]
    n_intervals = len(breaks) - 1
    timegap = breaks[1:] - breaks[:-1]
    breaks_midpoint = breaks[:-1] + 0.5 * timegap
    y_train = np.zeros((n_samples, n_intervals * 2))
    for i in range(n_samples):
        if f[i]:  # if failed (not censored)
            y_train[i, 0:n_intervals] = 1.0 * (t[i] >= breaks[
                                                       1:])  # give credit for surviving each time interval where failure time >= upper limit
            if t[i] < breaks[
                -1]:  # if failure time is greater than end of last time interval, no time interval will have failure marked
                y_train[i, n_intervals + np.where(t[i] < breaks[1:])[0][
                    0]] = 1  # mark failure at first bin where survival time < upper break-point
        else:  # if censored
            y_train[i, 0:n_intervals] = 1.0 * (t[i] >= breaks_midpoint)  # if censored and lived more than half-way through interval, give credit for surviving the interval.
    return y_train


def LDataSimu(seed = 1234, sampleSize=500, max_time=30, simu_dim=10, scale=1,overlap=1, plot=False, std = 1, confound = 0.2):
    np.random.seed(seed)

    scale = np.int(scale)

    # Define seed functions
    def x_seed(simu_dim, sampleSize):
        first_cut = int(np.round(simu_dim / 2))
        this_x = np.zeros((sampleSize, simu_dim), dtype=np.float32)
        for i in range(0, first_cut):
            this_x[:, i] = abs(np.random.normal(0, std, size=sampleSize))
        # this_x = np.array(list(map(lambda x: (x - x.min()) / (x.max() - x.min()), this_x)))
        for i in range(first_cut+1, simu_dim):
            this_x[:, i] = np.random.choice(range(0, 2), size=sampleSize)

        return this_x

    def treatment_assignment(seed_x, overlap):
        P = np.sum(seed_x[:, 0:3], axis=1) >  (np.median(np.sum(seed_x[:, 0:3], axis=1)))
        P = (1-overlap) * P + (overlap) * 0.5
        A =  np.array([np.random.choice([0,1], 1, p = [1-p, p]) for p in P]).reshape(-1)

        return A

    def x_t(t, x_i):
        cut = int(np.round(len(x_i) / 2))
        if t == 0:
            t_effect = 1
        else:
            t_effect = np.exp(-(t ** 0.5) * np.log10(t) + t ** 0.5 + 0.013 * t ** 1.5)
        x_t = x_i.copy()
        x_t[0:cut] = x_t[0:cut] / t_effect
        return x_t

    def surv_func_wrapper(train_x, A, sampleSize, t_start=0, max_time=max_time, plot=False, confounding = confound):
        train_x = np.array(train_x)
        A = np.array(A)

        def hazard(t, x_i, a_i, lamda_t=max_time * scale):
            beta0 = 1.0
            beta_e_1 = confounding
            beta_e_2 = confounding
            firstcut = int(np.round(x_i.shape[0]) / 2)
            x = x_i  # x_t(t, x_i)
            if t == 0:
                baseline = 0.1
            else:
                baseline = np.log10(t)
            LMD = (beta0 * a_i +
                   beta_e_1 * np.sum(x[0: firstcut]) +
                   beta_e_2 * np.sum(x[firstcut + 1: simu_dim])) / lamda_t
            return LMD * baseline

        def surv_func(t, x_i=None, a_i=None):
            x_i = np.array(x_i).reshape(-1)
            lmd = hazard(t, x_i, a_i)
            return np.exp(- lmd)

        def get_truesurv(t):
        #for t in tqdm(range(max(0, t_start - 1), max_time+3)):
            if t == t_start:
                list = Parallel(n_jobs=1)(delayed(surv_func)(1, train_x[(train_x[:,0] == i) & (train_x[:,1] == t), 3:], A[(train_x[:,0] == i)& (train_x[:,1] == t) ]) for i in np.unique(train_x[:,0])  )  # 273
                cSurv = np.array(list).reshape((1, sampleSize))  # 1-CDF
            else:
                list = Parallel(n_jobs=1)(delayed(surv_func)(t, train_x[(train_x[:,0] == i) & (train_x[:,1] == t), 3:], A[(train_x[:,0] == i)& (train_x[:,1] == t) ]) for i in np.unique(train_x[:,0])  )  # 273
                cSurv = np.array(list).reshape((1, sampleSize))  # 1-CDF
            #trueSurv = np.append(trueSurv, cSurv, axis=0)
            return cSurv

        trueSurv = Parallel(n_jobs=-1)(delayed(get_truesurv)(t) for t in (range(max(0, t_start - 1), max_time+3)))
        trueSurv = np.concatenate(trueSurv)
        trueSurv = trueSurv.T[:, 1:]

        if plot:
            plt.plot(trueSurv.T, alpha = 0.1, color = "#8DBFC5")
            plt.show()

        def get_truesurv_1(t, arm=1):
            if t == t_start:
                list = Parallel(n_jobs=1)(delayed(surv_func)(1, train_x[(train_x[:,0] == i) & (train_x[:,1] == t), 3:], arm ) for i in np.unique(train_x[:,0])  )
                cSurv = np.array(list).reshape((1, sampleSize))  # 1-CDF
            else:
                list = Parallel(n_jobs=1)(delayed(surv_func)(t, train_x[(train_x[:,0] == i) & (train_x[:,1] == t), 3:], arm ) for i in np.unique(train_x[:,0])  )
                cSurv = np.array(list).reshape((1, sampleSize))  # 1-CDF
            return cSurv
        trueSurv_1 = Parallel(n_jobs=-1)(delayed(get_truesurv_1)(t,1) for t in (range(max(0, t_start - 1), max_time + 3)))
        trueSurv_1 = np.concatenate(trueSurv_1)
        trueSurv_1 = trueSurv_1.T[:, 1:]

        trueSurv_0 = Parallel(n_jobs=-1)(delayed(get_truesurv_1)(t, 0) for t in (range(max(0, t_start - 1), max_time + 3)))
        trueSurv_0 = np.concatenate(trueSurv_0)
        trueSurv_0 = trueSurv_0.T[:, 1:]

        return trueSurv, trueSurv_1, trueSurv_0

    def cens_func_wrapper(train_x, A, sampleSize,t_start=0, max_time=max_time, scale=20):
        train_x = np.array(train_x)
        A = np.array(A)
        def censor_hazard(t, x_i):
            beta_e_1 = 0.1
            x = x_i  # x_t(t, x_i)
            if t == 0:
                baseline = 0.1
            else:
                baseline = np.log10(t)
            LMD = beta_e_1 * np.sum(x[0: 2])
            return LMD * baseline / scale

        def censor_func(t, x_i, a_i):
            x_i = np.array(x_i).reshape(-1)
            lmd = censor_hazard(t, x_i)
            return np.exp(- lmd)

        def get_truesurv(t):
            # for t in tqdm(range(max(0, t_start - 1), max_time+3)):
            if t == t_start:
                list = Parallel(n_jobs=1)(
                    delayed(censor_func)(1, train_x[(train_x[:, 0] == i) & (train_x[:, 1] == t), 3:],
                                       A[(train_x[:, 0] == i) & (train_x[:, 1] == t)]) for i in
                    np.unique(train_x[:, 0]))  # 273
                cSurv = np.array(list).reshape((1, sampleSize))  # 1-CDF
            else:
                list = Parallel(n_jobs=1)(
                    delayed(censor_func)(t, train_x[(train_x[:, 0] == i) & (train_x[:, 1] == t), 3:],
                                       A[(train_x[:, 0] == i) & (train_x[:, 1] == t)]) for i in
                    np.unique(train_x[:, 0]))  # 273
                cSurv = np.array(list).reshape((1, sampleSize))  # 1-CDF
            # trueSurv = np.append(trueSurv, cSurv, axis=0)
            return cSurv

        trueCens = Parallel(n_jobs=-1)(delayed(get_truesurv)(t) for t in (range(max(0, t_start - 1), max_time + 3)))
        trueCens = np.concatenate(trueCens)
        trueCens = trueCens.T[:, 1:]
        return trueCens


    #print('Generate seed data at time 0')
    seed_x = x_seed(simu_dim, sampleSize)
    A = treatment_assignment(seed_x, overlap)
    train_x = [np.array([np.append([i, t, A[i]], x_t(t, seed_x[i])) for t in range(0, max_time+7)]) for i in
               range(0, sampleSize) ]
    train_x = np.concatenate(train_x)

    train_x = pd.DataFrame(train_x)
    train_x.columns = np.arange(0,len( train_x.columns )).astype(str)
    train_x = train_x.rename(columns={"1": "T", "2": "A"})

    trueSurv, trueSurv_1, trueSurv_0 = surv_func_wrapper(train_x, train_x.A, sampleSize, plot=False)
    trueCens = cens_func_wrapper(train_x,train_x.A, sampleSize, max_time=max_time+3, scale = 30)

    #print('Simulate failure time')
    trueT = np.ones((sampleSize)) * max_time
    trueC = np.ones((sampleSize)) * max_time
    event = np.zeros((sampleSize))
    censor = np.zeros((sampleSize))

    for t in range(0, max_time+1):
        # T = np.array([integrate.quad(surv_func_t, t-1, t, args=(x_i, a_i))[0] for x_i, a_i in zip(seed_x, A)])
        T = trueSurv[:, t] < np.random.uniform(0, 1, sampleSize)
        trueT[(T == True) & (event == 0)] = t
        event[T == True] = 1
        C =  trueCens[:, t] < np.random.uniform(0, 1, sampleSize)
        trueC[(C == True) & (censor == 0)] = t
        censor[C == True] = 1

    time = np.minimum(trueT, trueC)
    event = (time == trueT) & (event == 1)
    time = np.round(time, 0)

    X = train_x.reset_index(drop = True)
    idx_censor = []
    for id in np.unique(X['0']):
        temp =  X.loc[X['0'] == id]['T']
        temp_idx = temp.index <= temp[temp <= time[int(id)]].index[-1]
        temp_idx = temp.index[temp_idx]
        temp_idx = np.array(temp_idx)
        idx_censor.append(temp_idx)
    idx_censor = np.concatenate(idx_censor)
    X = X.iloc[idx_censor].reset_index(drop = True)
    A = pd.DataFrame(np.array(X['A']), columns = ['A']).reset_index(drop = True)
    T = pd.DataFrame(np.array(X['T']), columns = ['T']).reset_index(drop = True)
    df = X.copy()
    df['Y'] = 0
    for i in np.unique(df['0']):
        temp = np.zeros(len( df.loc[df['0'] == i,'Y']))
        temp[-1] =  event[int(i)] * 1
        df.loc[df['0'] == i,'Y'] = temp


    if plot:
        plt.plot(range(trueSurv[A == 1].shape[1]), np.mean(np.cumprod(trueSurv[A == 1], 1), 0), '--', color="#448396",
                 label="ATT")
        plt.plot(range(trueSurv[A == 0].shape[1]), np.mean(np.cumprod(trueSurv[A == 0], 1), 0), '--', color="#448396",
                 label="ATC")
        plt.plot(range(trueSurv.shape[1]), np.mean(np.cumprod(trueSurv, 1), 0), '-', color="#666396", label='average')
        plt.plot(range(trueSurv.shape[1]), np.mean(np.cumprod(trueSurv_1, 1), 0), '-', color="#448396",
                 label='True treated')
        plt.plot(range(trueSurv.shape[1]), np.mean(np.cumprod(trueSurv_0, 1), 0), '-', color="#448396",
                 label='True control')

        kmf = KaplanMeierFitter()
        kmf.fit(time, event_observed=event * 1.)
        plt.plot(kmf.survival_function_.index.values, kmf.survival_function_.KM_estimate, '.', color='k', label="KM")
        plt.plot(range(trueSurv.shape[1]), np.mean(np.cumprod(trueSurv, 1), 0), '-', color="#448396")
        plt.xlim([0, 30])
        plt.ylim([0, 1])
        plt.legend()
        plt.show()
        plt.hist(time)
        plt.show()


    #print('step 1 done!')


    return df, train_x, surv_func_wrapper


def build_data_surv_rnn(train, score = None, history_itvl=14, prediction_itvl=7, max_time=30):
    observation = train['0']
    one_X = train.groupby(train['0'])['T'].max()
    event = train.groupby(train['0'])['Y'].max()
    train.drop(['T', 'Y', '0'], axis=1, inplace=True)
    train_x = np.array(train.reset_index(drop=True))
    x_dim = train_x.shape[1]

    f = event == 1

    def the_loop(i):
        rnn_y = np.empty((0, max_time * 2), dtype=np.float32)
        rnn_x = np.empty((0, history_itvl, x_dim), dtype=np.float32)
        rnn_m = np.empty((0, history_itvl, x_dim), dtype=np.float32)
        rnn_s = np.empty((0, history_itvl, x_dim), dtype=np.float32)


        ID = np.empty((0), dtype=np.float32)
        Time = np.empty((0), dtype=np.float32)
        Event = np.empty((0), dtype=np.float32)

        max_engine_time = int(one_X[i])
        yobs = np.zeros((2, max_time), dtype=np.float32)

        if score is not None:
            rnn_utility = np.empty((0, max_time, 3), dtype=np.float32)
            utility_obs = np.zeros((1, max_time, 3), dtype=np.float32)
            utility_obs[:,0:max_engine_time+1, :] = score[observation == i, :]
        else:
            rnn_utility = np.empty((0, max_time, 3), dtype=np.float32)
            utility_obs = np.zeros((1, max_time, 3), dtype=np.float32)

        if f[i]:
            yobs[0, 0:max_engine_time] = 1.0
            try:
                yobs[1, max_engine_time ] = 1.0
            except:
                pass
        else:
            yobs[0, 0:max_engine_time+1] = 1.0

        start = max(0, max_engine_time - prediction_itvl)
        end = max(max_engine_time  , 2)
        step = 1
        count = 0
        for j in np.arange(end, start, -step):
            covariate_x = train_x[observation == i,:]
            #covariate_x[0,:] = covariate_x[1,:]
            xtemp = np.ones((1, history_itvl, x_dim), dtype=np.float32)*-1
            x_end = min(j+1, max_engine_time)
            x_start = max(x_end-history_itvl, 0)
            x_end_t = covariate_x[x_start:x_end,:].reshape((1, -1, x_dim)).shape
            xtemp[0,0:x_end_t[1], : ] = covariate_x[x_start:x_end,:].reshape((1, -1, x_dim))


            ytemp = np.zeros((1, max_time * 2))
            y_end = min(j + 1, max_engine_time + 1)
            yobs_temp = yobs.copy()
            yobs_temp[0, y_end:] = 0
            yobs_temp[1, y_end:] = 0
            ytemp[0, 0:max_time] = yobs_temp[0, 0:max_time]
            ytemp[0, max_time:] = yobs_temp[1, 0:max_time]

            if sum(ytemp[0, prediction_itvl:prediction_itvl * 2]) > 0:
                _event = np. array(1).reshape(1)
            else:
                _event = np.array(0).reshape(1)

            x_temp, m_temp, s_temp = arr_to_xmd(xtemp)

            rnn_y = np.concatenate((rnn_y, ytemp))
            rnn_x = np.concatenate((rnn_x, x_temp))
            rnn_m = np.concatenate((rnn_m, m_temp))
            rnn_s = np.concatenate((rnn_s, s_temp))
            rnn_utility =  np.concatenate((rnn_utility, utility_obs))
            ID = np.concatenate((ID, np.array(i).reshape(1)))  # np.array(i).reshape(1)  #
            Time = np.concatenate((Time, np.array(count).reshape(1)))  # np.array(j).reshape(1)  #
            Event = np.concatenate((Event, _event))  # _event  #
            count = count+1


        return rnn_y, rnn_x, rnn_m, rnn_s, ID, Time, Event,rnn_utility

    the_list = Parallel(n_jobs=-1)(delayed(the_loop)(i) for i in set(observation))

    rnn_y = np.concatenate([item[0] for item in the_list])
    rnn_x = np.concatenate([item[1] for item in the_list])
    rnn_m = np.concatenate([item[2] for item in the_list])
    rnn_s = np.concatenate([item[3] for item in the_list])

    ID = np.concatenate([item[4] for item in the_list])
    Time = np.concatenate([item[5] for item in the_list])
    Event = np.concatenate([item[6] for item in the_list])
    rnn_utility = np.concatenate([item[7] for item in the_list])

    assert rnn_m.shape[0] == rnn_y.shape[0], 'Output dimension not match'
    assert ID.shape[0] == rnn_y.shape[0], 'Output dimension not match'
    assert Time.shape[0] == rnn_y.shape[0], 'Output dimension not match'
    assert Event.shape[0] == rnn_y.shape[0], 'Output dimension not match'


    return rnn_x, rnn_m, rnn_s, rnn_y, ID, Time, Event, rnn_utility


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
            'Initialization'
            self.rnn_x, self.rnn_m, self.rnn_s, self.rnn_y, _= data
            self.batch_size = batch_size
            self.shuffle = shuffle

            self.rnn_x0 = self.rnn_x.copy()
            self.rnn_x0[:, :, 0] = 0
            self.rnn_x1 = self.rnn_x.copy()
            self.rnn_x1[:, :, 0] = 1

            self.rnn_m0 = self.rnn_m.copy()
            self.rnn_m1 = self.rnn_m.copy()
            self.rnn_m0[self.rnn_x[:, :, 0]==0] = -1
            self.rnn_m1[self.rnn_x[:, :, 0]==1] = -1

            self.on_epoch_end()

        def __len__(self):
            'Denotes the number of batches per epoch'
            return int(np.floor(len(self.rnn_y) / self.batch_size))

        def __getitem__(self, index):
            'Generate one batch of data'
            # Generate indexes of the batch

            indexes1 = self.indexes1[index * self.batch_size1:(index + 1) * self.batch_size1]
            indexes0 = self.indexes0[index * self.batch_size0:(index + 1) * self.batch_size0]

            # Generate data
            X, y = self.__data_generation(indexes1, indexes0)

            return X, y

        def on_epoch_end(self):
            'Updates indexes after each epoch'
            self.indexes = np.arange(len(self.rnn_x))
            self.indexes0 = self.indexes[(self.rnn_x[:, 0:1, 0] == 0).reshape(-1)]
            self.indexes1 = self.indexes[(self.rnn_x[:, 0:1, 0] == 1).reshape(-1)]
            self.indexes0_len = len(self.indexes0 )
            self.indexes1_len = len(self.indexes1 )
            self.batch_size1 = int(self.batch_size * (self.indexes1_len / (self.indexes0_len + self.indexes1_len)))
            self.batch_size0 = int(self.batch_size * (self.indexes0_len / (self.indexes0_len + self.indexes1_len)))

            if self.shuffle == True:
                np.random.shuffle(self.indexes0)
                np.random.shuffle(self.indexes1)


        def __data_generation(self, indexes1, indexes0):
            'Generates data containing batch_size samples'  # X : (n_samples, *dim, n_channels)
            # Initialization
            index = np.concatenate([indexes1, indexes0])

            X = [self.rnn_x[index,:, 1:], self.rnn_m[index,:, 1:],
                 self.rnn_x[index], self.rnn_m[index], self.rnn_s[index],
                 self.rnn_x0[index], self.rnn_x1[index],
                 self.rnn_m0[index], self.rnn_m1[index]]

            #y_temp = np.concatenate([self.rnn_y[index], self.rnn_x[index, 0:1, 0]], axis=1)
            y = np.concatenate([self.rnn_y[index],self.rnn_y[index]], axis=1 )
            y[self.rnn_x[index,0:1,0].reshape(-1) == 0, 40: ] = -10
            y[self.rnn_x[index,0:1,0].reshape(-1) == 1, 0:40] = -10

            return X, y


class DataGenerator_p(tf.keras.utils.Sequence):
    def __init__(self, data, batch_size=32, shuffle=True):
        'Initialization'
        self.rnn_x, self.rnn_m, self.rnn_s, self.rnn_y, _ = data
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.on_epoch_end()

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(len(self.rnn_y) / self.batch_size))

    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch

        indexes1 = self.indexes1[index * self.batch_size1:(index + 1) * self.batch_size1]
        indexes0 = self.indexes0[index * self.batch_size0:(index + 1) * self.batch_size0]

        # Generate data
        X, y = self.__data_generation(indexes1, indexes0)

        return X, y

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(len(self.rnn_x))
        self.indexes0 = self.indexes[(self.rnn_x[:, 0:1, 0] == 0).reshape(-1)]
        self.indexes1 = self.indexes[(self.rnn_x[:, 0:1, 0] == 1).reshape(-1)]
        self.indexes0_len = len(self.indexes0)
        self.indexes1_len = len(self.indexes1)
        self.batch_size1 = int(self.batch_size * (self.indexes1_len / (self.indexes0_len + self.indexes1_len)))
        self.batch_size0 = int(self.batch_size * (self.indexes0_len / (self.indexes0_len + self.indexes1_len)))

        if self.shuffle == True:
            np.random.shuffle(self.indexes0)
            np.random.shuffle(self.indexes1)

    def __data_generation(self, indexes1, indexes0):
        'Generates data containing batch_size samples'  # X : (n_samples, *dim, n_channels)
        # Initialization
        index = np.concatenate([indexes1, indexes0])

        X = [self.rnn_x[index,:, 1:], self.rnn_m[index, :, 1:]]
        y = np.concatenate([self.rnn_y[index], self.rnn_x[index, 0:1, 0]], axis=1)

        return X, y
