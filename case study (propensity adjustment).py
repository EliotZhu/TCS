import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import tensorflow as tf
from joblib import Parallel, delayed
from scipy.optimize import minimize
from sklearn.linear_model import LinearRegression

from utilss.model import create_model, get_counterfactuals, benchmark_algorithms
from utilss.Evaluations import custom_auc, get_concordance
from utilss.data_simulator import get_data

from lifelines import CoxTimeVaryingFitter
from lifelines.utils import add_covariate_to_timeline
from lifelines.utils import to_long_format


print("simulating..." )
max_time = 30
history_itvl = 14

def ate_experiment(overlap):
    data, val_data, test_data, train_stat, val_stat, test_stat,surv_func_wrapper, train_full, val_full, test_full, raw= \
    get_data(input_dim= 5 , sampleSize= 1500, max_time=max_time,prediction_itvl = 1, history_itvl=14, overlap=overlap,
             seed= np.random.random_integers(1,1000), std = 1)
    print("simulation completed" )


    trueSurv_wrapper = surv_func_wrapper(train_full, train_full.A, len(train_stat), t_start=0, max_time=max_time,
                                         plot=False)

    s_true = np.cumprod(trueSurv_wrapper[0],1)
    cf_true = np.cumprod(trueSurv_wrapper[2],1) - np.cumprod(trueSurv_wrapper[1],1)
    hr_true = trueSurv_wrapper[2] / trueSurv_wrapper[1]


    # Train the model
    rnn_x, rnn_m, rnn_s, rnn_y, Time = data #estimation data set
    rnn_x_test, rnn_m_test, rnn_s_test, rnn_y_test, Time_test = test_data
    rnn_x_val, rnn_m_val, rnn_s_val, rnn_y_val, Time_val = val_data


    # Model fitting lstm model with censor loss
    modelCDSM,model_p, history_dict = create_model(rnn_x.shape[2], max_time, history_itvl, data, val_data, lstm_window= 7,
                                            alpha= 1, beta= 1, gamma=1.2, load=False, verbose=0, model_name='cox',
                                            batch_size= 256, layers=3)

    modelCDSM = get_counterfactuals(modelCDSM, data, t=0, draw=10, test_data=test_data)

    y_pred_t_adj, y_pred_std_adj, y_pred1_t_adj, y_pred0_t_adj, cf_std_1_adj, _, \
    y_pred_t_test_adj, y_pred_std_test_adj, y_pred1_t_test_adj, y_pred0_t_test_adj, cf_std_1_test_adj, _ = modelCDSM
    s_cdsm = np.cumprod(y_pred_t_adj, 1)
    cf_cdsm = np.cumprod(y_pred1_t_adj, 1) - np.cumprod(y_pred0_t_adj, 1)
    hr_cdsm = np.clip(y_pred1_t_adj, 0.001, 1) / np.clip(y_pred0_t_adj, 0.001, 1)

    #############################################
    modelCDSM_na,model_p, history_dict = create_model(rnn_x.shape[2], max_time, history_itvl, data, val_data, lstm_window= 7,
                                            alpha= 1, beta= 1, gamma=0, load=False, verbose=0, model_name='cox',
                                            batch_size= 256, layers=3)

    modelCDSM_na = get_counterfactuals(modelCDSM_na, data, t=0, draw=10, test_data=test_data)


    y_pred_t, y_pred_std, y_pred1_t, y_pred0_t, cf_std_1, _, \
    y_pred_t_test, y_pred_std_test, y_pred1_t_test, y_pred0_t_test, cf_std_1_test, _ = modelCDSM_na
    s_durv = np.cumprod(y_pred_t, 1)
    cf_durv = np.cumprod(y_pred1_t, 1) - np.cumprod(y_pred0_t, 1)
    hr_durv = np.clip(y_pred1_t, 0.001, 1) / np.clip(y_pred0_t, 0.001, 1)


    #############################################
    #IPW
    propensity_cdsm = np.array(model_p([rnn_x[:,:,1:], rnn_m[:,:,1:]])[:,0]).reshape(-1,1)
    weight1 = rnn_x[Time == 0,0,0].reshape(-1,1) / np.clip(propensity_cdsm[Time == 0],a_min = 0.05, a_max=1)
    weight0 = (1-rnn_x[Time == 0,0,0].reshape(-1,1)) / np.clip(1-propensity_cdsm[Time == 0],a_min = 0.05, a_max=1)

    cf_durv_IPW = np.mean(np.cumprod(y_pred_t, 1) * weight1 * sum(rnn_x[Time == 0,0,0])/len(y_pred_t) -\
                  np.cumprod(y_pred_t, 1) * weight0 * (len(y_pred_t)-sum(rnn_x[Time == 0,0,0]))/len(y_pred_t),0)
    hr_durv_IPW = np.mean(np.cumprod(y_pred_t, 1)* weight1 ,0) / np.mean(np.cumprod(y_pred_t, 1)* weight0 ,0)

    #############################################
    #TMLE
    propensity_cdsm = np.array(model_p([rnn_x[:,:,1:], rnn_m[:,:,1:]])[:,0]).reshape(-1,1)
    y_pred_t_tmle_1 = y_pred1_t.copy()
    y_pred_t_tmle_0 = y_pred0_t.copy()

    weight1 = rnn_x[Time == 0,0,0].reshape(-1,1)
    weight0 = (1-rnn_x[Time == 0,0,0].reshape(-1,1))
    H1 = 1 / np.clip(propensity_cdsm[Time == 0],a_min = 0.01, a_max=1)
    H0 = 1 / np.clip(1-propensity_cdsm[Time == 0],a_min = 0.01, a_max=1)
    HA = weight1*H1 - weight0*H0

    deltas = []
    for i in range(y_pred_t.shape[1]):
        reg = LinearRegression().fit(np.concatenate([HA,y_pred_t[:,i].reshape(-1,1)],1), rnn_y[Time == 0][:,i])
        deltas.append(reg.coef_[0])
    for i in range(y_pred_t.shape[1]):
        y_pred_t_tmle_1[:,i] = y_pred1_t[:,i] + deltas[i] * H1.reshape(-1)
        y_pred_t_tmle_0[:,i] = y_pred0_t[:,i] + deltas[i] * H0.reshape(-1)

    cf_durv_tmle =  np.mean(np.cumprod(y_pred_t_tmle_1, 1)* weight1 - np.cumprod(y_pred_t_tmle_0, 1)* weight0 ,0)
    hr_durv_tmle = np.mean(np.cumprod(y_pred_t_tmle_1, 1)* weight1 ,0) / np.mean(np.cumprod(y_pred_t_tmle_0, 1)* weight0 ,0)

    #############################################

    # sns.set(style="whitegrid", font_scale=1)
    # fig, ax = plt.subplots(figsize=(7, 7))
    # ax.plot(range(max_time), np.mean(cf_true[:, 0:max_time], 0), color="#8c8c8c", label="True")
    # ax.plot(range(max_time), cf_durv_IPW, '--', color="#f0aeb4", label="IPW")
    # ax.plot(range(max_time), cf_durv_tmle, '.', color="#f0aeb4", label="TMLE")
    # ax.plot(range(max_time), np.mean(cf_durv, 0), color='#8DBFC5', alpha=0.9, label="CDSM")
    # ax.plot(range(max_time), np.mean(cf_cdsm, 0), '+', color='#8DBFC5', alpha=1, label="CDSM (NC)")
    # # ax.set_xticklabels(np.arange(-8, max_time * 3, 8))
    # ax.set_xlabel("Time", fontsize=11, fontweight='bold')
    # ax.set_ylabel("ATE (Difference in Survival Probability)", fontsize=11, fontweight='bold')
    # plt.legend()
    # plt.savefig("plots/ate_causal.png", bbox_inches='tight', pad_inches=0.5, dpi=500)
    # plt.show()

    #############################################

    cv_df = raw.copy()
    base_df = to_long_format(cv_df[['0', 'T', 'Y']], duration_col="T")
    base_df = add_covariate_to_timeline(base_df, cv_df[["A", "3","4", '0', 'T', 'Y']], duration_col="T", id_col="0",
                                        event_col="Y").fillna(0)
    base_df = base_df.drop_duplicates().reset_index(drop=True)

    base_df = base_df.loc[~((base_df["start"] == base_df["stop"]) & (base_df["start"] == 0))]

    ctv = CoxTimeVaryingFitter(penalizer=0.1)
    ctv.fit(base_df, id_col="0", event_col="Y", start_col="start", stop_col="stop", show_progress=True)
    hr_cox = ctv.params_.A

    #############################################
    #compose the result

    true_ate = np.mean(cf_true[:,0:max_time], 0)
    true_ate[0] = 0.001
    ate_bias = np.concatenate([(np.abs(np.mean(cf_cdsm, 0) - true_ate)/true_ate).reshape(-1,1),
                               (np.abs(np.mean(cf_durv, 0) - true_ate)/true_ate).reshape(-1,1),
                               (np.abs(cf_durv_IPW - true_ate)/true_ate).reshape(-1,1),
                               (np.abs(cf_durv_tmle - true_ate)/true_ate).reshape(-1,1)],1)
    ate_bias_selected = ate_bias[[1, 15, 29],:]


    hr_true_average = np.mean(hr_true[:,0:max_time])
    hr_dsurv_average = np.mean(hr_durv)
    hr_cdsm_average = np.mean(hr_cdsm)
    hr_durv_IPW_average  = np.mean(hr_durv_IPW)
    hr_durv_tmle_average = np.mean(hr_durv_tmle)
    hr_cox = np.exp(hr_cox)

    hr_bias = np.array([(hr_cdsm_average - hr_true_average)/hr_true_average,
               (hr_dsurv_average - hr_true_average) / hr_true_average,
               (hr_durv_IPW_average - hr_true_average) / hr_true_average,
               (hr_durv_tmle_average - hr_true_average) / hr_true_average,
               (hr_cox - hr_true_average) / hr_true_average]).reshape(1,5)

    return ate_bias_selected, hr_bias

ate_bias = []
hr_bias = []
for i in range(10):
    ate_bias_one, hr_bias_one = ate_experiment(1)
    ate_bias.append(ate_bias_one)
    hr_bias.append(hr_bias_one)

ate_bias_avg = np.mean(np.concatenate([ate_bias]),0)
ate_bias_std = np.std(np.concatenate([ate_bias]),0)

hr_bias_avg = np.mean(np.concatenate([hr_bias]),0)
hr_bias_std = np.std(np.concatenate([hr_bias]),0)



































