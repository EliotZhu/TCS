# data_simulator.py
#
# Author: Jie Zhu
# Tested with Python version 3.8 and TensorFlow 2.0

import numpy as np
from utilss.data_handler import LDataSimu, build_data_surv_rnn

##########################################################
# Data Simulation
def get_data(input_dim=10, sampleSize=1000, max_time=30, prediction_itvl=1, history_itvl=14, overlap=1,  seed=1123, std = 1):
    df, df_full, surv_func_wrapper = LDataSimu(sampleSize=sampleSize, max_time=max_time,
                                               simu_dim=input_dim, scale= 2, overlap = overlap,
                                               seed=np.random.seed(seed), plot=False, std = std)
    id_set = np.unique(df['0'])
    train_id, validate_id, test_id = np.split(id_set, [int(.7 * len(id_set)), int(.9 * len(id_set))])
    train_idx = df['0'].isin(train_id)
    validate_idx =  df['0'].isin(validate_id)
    test_idx = df['0'].isin(test_id)
    train = df[train_idx]
    val = df[validate_idx]
    test = df[test_idx]

    train_idx =  df_full['0'].isin(train_id)
    validate_idx = df_full['0'].isin(validate_id)
    test_idx = df_full['0'].isin(test_id)
    train_full = df_full[train_idx]
    val_full = df_full[validate_idx]
    test_full = df_full[test_idx]


    def preprocess_data(train):
        #np.sum(train.isna(),0)/len(train)
        one_X = train[train['T'] == 1].reset_index(drop=True)
        one_X = one_X.drop(['A', 'T', 'Y'], axis = 1)
        event = train.groupby('0')['Y'].max()
        A = train.groupby('0')['A'].max()
        time = train.groupby('0')[['Y', 'T']].apply(lambda x: min(x['T'][x.Y == 1]) if sum(x.Y) > 0 else max(x['T']))
        one_X.insert(0, 'ID', one_X['0'].reset_index(drop=True))
        one_X.insert(1, 'T.tilde', time.reset_index(drop=True))
        one_X.insert(2, 'Delta', event.reset_index(drop=True))
        one_X.insert(3, 'A', A.reset_index(drop=True))
        one_X = one_X.drop(['0'], axis = 1)
        one_X = np.array(one_X)

        return one_X

    train_stat = preprocess_data(train.copy())
    val_stat = preprocess_data(val.copy())
    test_stat = preprocess_data(test.copy())

    rnn_x, rnn_m, rnn_s, rnn_y, ID, Time, Event, _= \
        build_data_surv_rnn(train.copy(), score = None, history_itvl=history_itvl, prediction_itvl=prediction_itvl, max_time=max_time)

    rnn_x_val, rnn_m_val, rnn_s_val, rnn_y_val, ID_val, Time_val, Event_val, _  = \
        build_data_surv_rnn(val.copy(), score=None, history_itvl=history_itvl, prediction_itvl=prediction_itvl, max_time=max_time)

    rnn_x_test, rnn_m_test, rnn_s_test, rnn_y_test, ID_test, Time_test, Event_test, _ = \
        build_data_surv_rnn(test.copy(), score=None, history_itvl=history_itvl, prediction_itvl=prediction_itvl, max_time=max_time)

    rnn_x[np.isnan(rnn_x)] = 0
    rnn_x_val[np.isnan(rnn_x_val)] = 0
    rnn_x_test[np.isnan(rnn_x_test)] = 0

    data = [rnn_x, rnn_m, rnn_s, rnn_y, Time]
    val_data = [rnn_x_val, rnn_m_val, rnn_s_val, rnn_y_val, Time_val]
    test_data = [rnn_x_test, rnn_m_test, rnn_s_test, rnn_y_test, Time_test]


    return data, val_data, test_data, train_stat, val_stat, test_stat,surv_func_wrapper, train_full, val_full, test_full,\
           train

