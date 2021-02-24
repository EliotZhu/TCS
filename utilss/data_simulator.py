# data_simulator.py
#
# Author: Jie Zhu
# Tested with Python version 3.8 and TensorFlow 2.0

import numpy as np

from utilss.data_handler import LDataSimu, build_data_surv_rnn


##########################################################
# Data Simulation
def get_data(input_dim=6, sampleSize=600, max_time=30, prediction_itvl=1,
             history_itvl=14, overlap=1, seed=123, std=0.01, confound=0.5, scale=2):
    df, df_full, surv_func_wrapper = LDataSimu(seed, sampleSize=sampleSize, max_time=max_time,
                                               simu_dim=input_dim, scale=scale, overlap=overlap,
                                               plot=False, std=std, confound=confound)

    sum(df.Y) / max(df['ID'])
    # scale = 2  35%
    # scale = 10  10%
    # scale = 20  5%
    # scale = 100  1%
    #scale = 500  0.05%


    id_set = np.unique(df['ID'])
    train_id, validate_id, test_id = np.split(id_set, [int(.8 * len(id_set)), int(.9 * len(id_set))])
    train_idx = df['ID'].isin(train_id)
    validate_idx =  df['ID'].isin(validate_id)
    train = df[train_idx]
    val = df[validate_idx]


    train_idx =  df_full['ID'].isin(train_id)
    validate_idx = df_full['ID'].isin(validate_id)
    test_idx = df_full['ID'].isin(test_id)
    train_full = df_full[train_idx]
    val_full = df_full[validate_idx]
    test_full = df_full[test_idx]


    rnn_x, rnn_m, rnn_s, rnn_y, ID, Time, Event, _= \
        build_data_surv_rnn(train.copy(), score = None, history_itvl=history_itvl, prediction_itvl=prediction_itvl, max_time=max_time)


    rnn_x_val, rnn_m_val, rnn_s_val, rnn_y_val, ID_val, Time_val, Event_val, _ = \
        build_data_surv_rnn(val.copy(), score=None, history_itvl=history_itvl, prediction_itvl=prediction_itvl,
                            max_time=max_time)

    rnn_x_pred, rnn_m_pred, rnn_s_pred, rnn_y_pred, ID_pred, Time_pred, Event_pred, _ = \
        build_data_surv_rnn(df.copy(), score=None, history_itvl=history_itvl, prediction_itvl=prediction_itvl, max_time=max_time)

    rnn_x[np.isnan(rnn_x)] = 0
    rnn_x_pred[np.isnan(rnn_x_pred)] = 0

    data = [rnn_x, rnn_m, rnn_s, rnn_y, Time]
    pred_data = [rnn_x_pred, rnn_m_pred, rnn_s_pred, rnn_y_pred, Time_pred]
    val_data = [ rnn_x_val, rnn_m_val, rnn_s_val, rnn_y_val, Time_val]

    def preprocess_data(train):
        #np.sum(train.isna(),0)/len(train)
        one_X = train[train['T'] == 1].reset_index(drop=True)
        one_X = one_X.drop(['A'], axis = 1)
        event = train.groupby('patnumber')['Y'].max()
        A = train.groupby('patnumber')['A'].max()
        time = train.groupby('patnumber')['Y', 'T'].apply(lambda x: min(x['T'][x.Y == 1]) if sum(x.Y) > 0 else max(x['T']))
        one_X.insert(0, 'ID', one_X['patnumber'].reset_index(drop=True))
        one_X.insert(1, 'T.tilde', time.reset_index(drop=True))
        one_X.insert(2, 'Delta', event.reset_index(drop=True))
        one_X.insert(3, 'A', A.reset_index(drop=True))
        one_X = np.array(one_X)
        return one_X

    index = df.groupby(['ID']).apply(lambda x: max(x.index))
    one_X = df.iloc[np.array(index)]

    return data, pred_data,val_data, surv_func_wrapper, train_full, val_full, test_full, one_X

