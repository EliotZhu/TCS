import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, roc_auc_score, roc_curve
from nevergrad.optimization import optimizerlib
import nevergrad as ng
inst = ng.p.Instrumentation
from scipy.optimize import minimize

def get_concordance(y_pred, y_true, y_true_event):
    # Rank loss
    # temp = y_pred
    # y_pred = temp
    R = np.dot(y_pred, np.transpose(y_true))  # N*N
    diag_R = np.reshape(np.diag(R), (-1, 1))  # N*1 shows how long one survive
    one_vector = np.ones(np.shape(diag_R))  # N*1
    R = np.dot(one_vector, np.transpose(diag_R)) - R  # r_{i}(T_{j}) - r_{j}(T_{j}) does column survive longer than row?
    R = np.transpose(R)  # r_{i}(T_{i}) - r_{j}(T_{i}) does row survive longer than column

    I2 = np.reshape(np.sum(y_true_event, axis=1), (-1, 1))  # N*1
    I2 = np.equal(I2, 1) * 1
    I2 = np.dot(one_vector, I2.reshape(1, -1))  # censored or event
    T2 = np.reshape(np.sum(y_true, axis=1), (-1, 1))  # N*1

    T = np.sign(np.dot(one_vector, np.transpose(T2)) - np.dot(T2, np.transpose(one_vector)))  # (Ti(t)>Tj(t)=1); N*N
    T = np.maximum(T, 0)
    T = I2 * T  # only remains T_{ij}=1 when event occured for subject i 1*N
    eta = T * np.exp(-R / 0.1)
    eta = T * (eta >= T) * 1.0
    concordance = np.sum(eta) / np.sum(T)

    return concordance

def custom_auc(y_pred, y_true, plot):
    #y_pred = np.cumprod(y_pred, axis=1)
    probas = y_pred.reshape(-1)
    preds = (probas > 0.5).astype(int)
    auc = roc_auc_score(y_true.reshape(-1), probas)

    fpr, tpr, _ = roc_curve(y_true.reshape(-1), probas)

    if plot:
        plt.plot(fpr, tpr, 'r-')
        plt.show()
    return auc

def custom_accuracy(y_pred, y_true):
    probas = y_pred.reshape(-1)
    preds = (probas > 0.5).astype(int)
    acc = accuracy_score(y_true.reshape(-1), preds)


    return acc



def get_distance(s_rnn_b, pred_rnn_y, pred_time, threshold, max_time):
    event = np.sum(pred_rnn_y[pred_time == 0, max_time:2 * max_time], axis=1)
    censor = np.sum(pred_rnn_y[pred_time == 0, 0: max_time], axis=1)
    idx1 = np.where(event >= 1)
    idx2 = np.where((event == 0) & (censor != max_time))
    set1 = np.concatenate([np.where(np.diff(pred_rnn_y[pred_time == 0, 0:max_time][idx1]) == -1)[1].reshape(-1, 1),
                           np.apply_along_axis(lambda x: np.min(
                               np.where(x <= threshold) if len(np.where(x <= threshold)[0]) > 0 else max_time), 1,
                                               s_rnn_b[idx1][:, :-1]).reshape(-1, 1)], axis=1)

    set2 = np.concatenate([np.where(np.diff(pred_rnn_y[pred_time == 0, 0:max_time][idx2]) == -1)[1].reshape(-1, 1),
                           np.apply_along_axis(lambda x: np.min(
                               np.where(x <= threshold) if len(np.where(x <= threshold)[0]) > 0 else max_time),
                                               1, s_rnn_b[idx2][:, :-1]).reshape(-1, 1)], axis=1)

    dist1 = np.mean(np.abs(set1[:, 0] - set1[:, 1]))
    dist2 = np.mean(np.abs(set2[:, 0] - set2[:, 1]))
    return (dist1 + dist2) / 2


def get_evaluation_true(model_result, rnn_y, rnn_y_test, Time, Time_test, max_time,
                   trueSurv_wrapper, trueSurv_wrapper_test):


    trueSurv, trueSurv_1, trueSurv_0 = trueSurv_wrapper
    trueSurv_t, trueSurv_1_t, trueSurv_0_t = trueSurv_wrapper_test
    y_pred_t, y_pred_std, y_pred1_t, y_pred0_t, cf_std_1, _, \
    y_pred_t_test, y_pred_std_test, y_pred1_t_test, y_pred0_t_test, cf_std_1_test, _ = model_result


    s_rnn_b = np.cumprod(y_pred_t, 1)
    cf_rnn_b = np.cumprod(y_pred0_t, 1) - np.cumprod(y_pred1_t, 1)

    s_rnn_b_test = np.cumprod(y_pred_t_test, 1)
    cf_rnn_b_test = np.cumprod(y_pred0_t_test, 1) - np.cumprod(y_pred1_t_test, 1)


    AUROC_b = custom_auc(s_rnn_b, rnn_y[Time == 0, 0:max_time], plot=False)
    concordance_b = get_concordance(s_rnn_b, rnn_y[Time == 0, 0:max_time],rnn_y[Time == 0, max_time:2 * max_time])

    AUROC_b_test = custom_auc(np.cumprod(y_pred_t_test, 1), rnn_y_test[Time_test == 0, 0:max_time], plot=False)
    concordance_b_test = get_concordance(np.cumprod(y_pred_t_test, 1), rnn_y_test[Time_test == 0, 0:max_time],
                                         rnn_y_test[Time_test == 0, max_time:2 * max_time])

    res = minimize(lambda x: get_distance(s_rnn_b, rnn_y, Time, max_time=max_time, threshold=x), 0.9,
                   method='nelder-mead', options={'xatol': 1e-7, 'disp': True})
    avg_dist = get_distance(s_rnn_b, rnn_y, Time, threshold=res.x, max_time=max_time)
    avg_dist_test = get_distance(np.cumprod(y_pred_t_test, 1), rnn_y_test, Time_test, threshold=res.x, max_time=max_time)

    dist_var = np.var(
        np.apply_along_axis(lambda x: get_distance(s_rnn_b, rnn_y, Time, threshold=x, max_time=max_time), 0,
                            np.arange(0.9, 1, 0.01).reshape(1, -1)))
    dist_var_test = np.var(
        np.apply_along_axis(lambda x: get_distance(s_rnn_b_test, rnn_y_test, Time_test, threshold=x, max_time=max_time),
                            0, np.arange(0.9, 1, 0.01).reshape(1, -1)))

    #When true is avaliable
    def get_rmse_bias_mean(s_rnn_b,cf_rnn_b, trueSurv, trueSurv_0, trueSurv_1):
        s_t = np.cumprod(trueSurv, 1)[:,:max_time]
        cf_t = np.cumprod(trueSurv_0, 1)[:,:max_time] - np.cumprod(trueSurv_1, 1)[:,:max_time]
        bias =   np.divide(np.abs(np.mean(s_rnn_b,0) - np.mean(s_t,0) +0.001) , np.mean(s_t,0)+0.001)
        bias_cf = np.divide(np.abs(np.mean(cf_rnn_b,0) - np.mean(cf_t,0) +0.001) , np.mean(cf_t,0)+0.001)
        return bias, bias_cf, bias, bias_cf

    def get_rmse_bias(s_rnn_b,cf_rnn_b, trueSurv, trueSurv_0, trueSurv_1):
        s_t = np.cumprod(trueSurv, 1)[:,:max_time]
        cf_t = np.cumprod(trueSurv_0, 1)[:,:max_time] - np.cumprod(trueSurv_1, 1)[:,:max_time]
        rmse = np.sqrt(np.mean(((s_rnn_b - s_t) / (1 + s_t)) ** 2, 0))
        rmse_cf = np.sqrt(np.mean(((cf_rnn_b - cf_t) / (cf_t + 1)) ** 2, 0))
        bias =  np.mean(  np.divide(np.abs(s_rnn_b - s_t +0.001) , s_t+0.001)   , 0 )
        bias_cf =  np.mean(  np.divide(np.abs(cf_rnn_b - cf_t +0.001) , cf_t+0.001)   , 0 )
        return rmse, rmse_cf, bias, bias_cf

    rmse, rmse_cf, bias, bias_cf = get_rmse_bias(s_rnn_b, cf_rnn_b, trueSurv, trueSurv_0, trueSurv_1)
    rmse_t, rmse_cf_t, bias_t, bias_cf_t = get_rmse_bias(s_rnn_b_test, cf_rnn_b_test, trueSurv_t, trueSurv_0_t, trueSurv_1_t)

    def subgroup_analyis(cf_rnn_b, s_rnn_b, trueSurv, trueSurv_0, trueSurv_1, y_pred_std, cf_std_1):
        try:
            s_t = np.cumprod(trueSurv, 1)[:, :max_time]
            cf_t = np.cumprod(trueSurv_0, 1)[:, :max_time] - np.cumprod(trueSurv_1, 1)[:, :max_time]
            cf_l = cf_rnn_b - 3.96 * cf_std_1 #/ np.sqrt(len(y_pred_std))
            cf_u = cf_rnn_b + 3.96 * cf_std_1 #/ np.sqrt(len(y_pred_std))
            s_l = s_rnn_b - 3.96 * y_pred_std #/ np.sqrt(len(y_pred_std))
            s_u = s_rnn_b + 3.96 * y_pred_std #/ np.sqrt(len(y_pred_std))

            # plt.plot(np.mean(s_t,0))
            # plt.plot(np.mean(s_rnn_b,0))
            # plt.plot(np.mean(s_u,0),'--')
            # plt.plot(np.mean(s_l,0),'--')
            # plt.show()

            coverage_cf_subB = []
            coverage_sf_subB = []
            for subgroup in [1, 5,10,20,40,80,160, len(y_pred_std)]:
                coverage_cf_sub = []
                coverage_sf_sub = []
                size = int(len(y_pred0_t) / subgroup)
                for i in range(size):
                    idx = np.random.choice(range(len(y_pred_std)), subgroup, replace= False)
                    coverage_cf_sub_tp = (np.mean(cf_t[idx], 0) >= np.mean(cf_l[idx], 0)) & (
                                np.mean(cf_t[idx], 0) <= np.mean(cf_u[idx], 0))
                    coverage_sf_sub_tp = (np.mean(s_t[idx], 0) >= np.mean(s_l[idx], 0)) & (
                                np.mean(s_t[idx], 0) <= np.mean(s_u[idx], 0))
                    coverage_cf_sub.append(coverage_cf_sub_tp.reshape(-1,1))
                    coverage_sf_sub.append(coverage_sf_sub_tp.reshape(-1,1))
                coverage_sf_sub = np.sum(np.concatenate(coverage_sf_sub,1),1)/size
                coverage_cf_sub = np.sum(np.concatenate(coverage_cf_sub,1),1)/size
                coverage_cf_subB.append( np.append(subgroup, coverage_cf_sub))
                coverage_sf_subB.append( np.append(subgroup, coverage_sf_sub))
        except:
            coverage_cf_subB = []
            coverage_sf_subB = []
            s_t = []
            cf_t = []
            cf_l = []
            cf_u = []
            s_l = []
            s_u = []

        try:
            rmse_cf_subB = []
            rmse_sf_subB = []
            bias_cf_subB = []
            bias_sf_subB = []
            for subgroup in [1, 5,10,20,40,80,160, len(cf_rnn_b)]:
                rmse_cf_sub = []
                rmse_sf_sub = []
                bias_cf_sub = []
                bias_sf_sub = []
                size = int(len(y_pred0_t) / subgroup)
                for i in range(size):
                    idx = np.random.choice(range(len(cf_rnn_b)), subgroup, replace= False)
                    rmse_tp, rmse_cf_tp, bias_tp, bias_cf_tp = get_rmse_bias_mean(s_rnn_b[idx], cf_rnn_b[idx], trueSurv[idx], trueSurv_0[idx], trueSurv_1[idx])
                    rmse_cf_sub.append(rmse_cf_tp.reshape(-1,1))
                    rmse_sf_sub.append(rmse_tp.reshape(-1,1))
                    bias_cf_sub.append(bias_cf_tp.reshape(-1, 1))
                    bias_sf_sub.append(bias_tp.reshape(-1, 1))

                rmse_cf_sub = np.mean(np.concatenate(rmse_cf_sub,1),1)
                rmse_sf_sub = np.mean(np.concatenate(rmse_sf_sub,1),1)
                bias_cf_sub = np.mean(np.concatenate(bias_cf_sub,1),1)
                bias_sf_sub = np.mean(np.concatenate(bias_sf_sub,1),1)

                rmse_cf_subB.append( np.append(subgroup, rmse_cf_sub))
                rmse_sf_subB.append( np.append(subgroup, rmse_sf_sub))
                bias_cf_subB.append( np.append(subgroup, bias_cf_sub))
                bias_sf_subB.append( np.append(subgroup, bias_sf_sub))

        except:
            rmse_cf_subB = []
            rmse_sf_subB = []
            bias_cf_subB = []
            bias_sf_subB = []
        return coverage_sf_subB, coverage_cf_subB, rmse_cf_subB, rmse_sf_subB, bias_cf_subB,bias_sf_subB, s_t, cf_t, cf_l, cf_u, s_l, s_u



    coverage_sf_subB_t, coverage_cf_subB_t, rmse_cf_subB_t, rmse_sf_subB_t, bias_cf_subB_t, bias_sf_subB_t, s_t, cf_t, cf_l, cf_u, s_l, s_u = \
        subgroup_analyis(cf_rnn_b_test, s_rnn_b_test, trueSurv_t, trueSurv_0_t, trueSurv_1_t, y_pred_std_test, cf_std_1_test)
    coverage_sf_subB, coverage_cf_subB, rmse_cf_subB, rmse_sf_subB, bias_cf_subB, bias_sf_subB, s_t, cf_t, cf_l, cf_u, s_l, s_u =  \
        subgroup_analyis(cf_rnn_b, s_rnn_b, trueSurv, trueSurv_0, trueSurv_1, y_pred_std, cf_std_1)

    def Compose_coverage(coverage_sf_subB, coverage_cf_subB, rmse_cf_subB, rmse_sf_subB, bias_cf_subB,bias_sf_subB):
        try:
            coverage_subgroup_sf = pd.DataFrame(coverage_sf_subB).loc[:, 1:].T
            coverage_subgroup_sf.columns = (pd.DataFrame(coverage_sf_subB).loc[:, 0]).astype('int32')
            coverage_subgroup_sf['group'] = 'coverage survival curve'

            coverage_subgroup_cf = pd.DataFrame(coverage_cf_subB).loc[:, 1:].T
            coverage_subgroup_cf.columns = (pd.DataFrame(coverage_cf_subB).loc[:, 0]).astype('int32')
            coverage_subgroup_cf['group'] = 'coverage causal effect'
        except:
            pass

        subgroup_rmse_cf = pd.DataFrame(rmse_cf_subB).loc[:, 1:].T
        subgroup_rmse_cf.columns = (pd.DataFrame(rmse_cf_subB).loc[:, 0]).astype('int32')
        subgroup_rmse_cf['group'] = 'rmse causal effect'


        subgroup_rmse_sf = pd.DataFrame(rmse_sf_subB).loc[:, 1:].T
        subgroup_rmse_sf.columns = (pd.DataFrame(rmse_sf_subB).loc[:, 0]).astype('int32')
        subgroup_rmse_sf['group'] = 'rmse survival curve'

        subgroup_bias_cf = pd.DataFrame(bias_cf_subB).loc[:, 1:].T
        subgroup_bias_cf.columns = (pd.DataFrame(bias_cf_subB).loc[:, 0]).astype('int32')
        subgroup_bias_cf['group'] = 'bias causal effect'

        subgroup_bias_sf = pd.DataFrame(bias_sf_subB).loc[:, 1:].T
        subgroup_bias_sf.columns = (pd.DataFrame(bias_sf_subB).loc[:, 0]).astype('int32')
        subgroup_bias_sf['group'] = 'bias survival curve'

        try:
            subgroup_df = pd.concat([coverage_subgroup_sf, coverage_subgroup_cf, subgroup_rmse_cf, subgroup_rmse_sf,
                                 subgroup_bias_cf, subgroup_bias_sf])
        except:
            subgroup_df = pd.concat([subgroup_rmse_cf, subgroup_rmse_sf, subgroup_bias_cf, subgroup_bias_sf])
        return  subgroup_df


    subgroup_df = Compose_coverage(coverage_sf_subB, coverage_cf_subB, rmse_cf_subB, rmse_sf_subB, bias_cf_subB, bias_sf_subB)
    subgroup_df_t = Compose_coverage(coverage_sf_subB_t, coverage_cf_subB_t, rmse_cf_subB_t, rmse_sf_subB_t, bias_cf_subB_t, bias_sf_subB_t)



    metrics = pd.DataFrame([AUROC_b,AUROC_b_test, concordance_b,concordance_b_test, avg_dist, avg_dist_test, dist_var, dist_var_test,
                            np.mean(rmse),  np.mean(rmse_t),  np.mean(rmse_cf),  np.mean(rmse_cf_t),  np.mean(bias) , np.mean(bias_t),
                            np.mean(bias_cf[15:30]),  np.mean(bias_cf_t[15:30])])
    metrics = metrics.T
    metrics.columns = ['AUROC','AUROC (Test)', 'Concordance','Concordance (Test)', 'Distance', 'Distance (Test)', 'Distance Std', 'Distance Std (Test)',
                       'RMSE', 'RMSE (Test)', 'RMSE Causal', 'RMSE Causal (Test)', 'Bias', 'Bias (Test)', 'Bias Causal', 'Bias Causal (Test)']

    try:
        metrics['Coverage'] = np.mean(subgroup_df[subgroup_df.group == 'coverage survival curve'][1])
        metrics['Coverage Causal)'] = np.mean(subgroup_df[subgroup_df.group == 'coverage causal effect'][1])
        metrics['Coverage (Test)'] = np.mean(subgroup_df_t[subgroup_df_t.group == 'coverage survival curve'][1])
        metrics['Coverage Causal (Test)'] = np.mean(subgroup_df_t[subgroup_df_t.group == 'coverage causal effect'][1])

    except:
        metrics['Coverage'] = 0
        metrics['Coverage Causal)'] = 0
        metrics['Coverage (Test)'] = 0
        metrics['Coverage Causal (Test)'] = 0


    return metrics,  subgroup_df,subgroup_df_t, s_rnn_b, cf_rnn_b, s_rnn_b_test, cf_rnn_b_test,s_t, cf_t, cf_l, cf_u, s_l, s_u



