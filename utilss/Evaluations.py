import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, roc_auc_score, roc_curve
from nevergrad.optimization import optimizerlib
import nevergrad as ng
inst = ng.p.Instrumentation
from scipy.optimize import minimize
import statsmodels.formula.api as smf

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
    idx2 = np.where((event == 0))# & (censor != max_time))
    try:
        set1 = np.concatenate([np.where(np.diff(pred_rnn_y[pred_time == 0, 0:max_time][idx1]) == -1)[1].reshape(-1, 1),
                               np.apply_along_axis(lambda x: np.min(
                                   np.where(x <= threshold) if len(np.where(x <= threshold)[0]) > 0 else max_time), 1,
                                                   s_rnn_b[idx1][:, :-1]).reshape(-1, 1)], axis=1)
        dist1 = np.mean(np.abs(set1[:, 0] - set1[:, 1]))
    except:
        dist1 = 1
    try:
        set2 = np.concatenate([np.where(np.diff(pred_rnn_y[pred_time == 0, 0:max_time][idx2]) == -1)[1].reshape(-1, 1),
                               np.apply_along_axis(lambda x: np.min(
                                   np.where(x <= threshold) if len(np.where(x <= threshold)[0]) > 0 else max_time),
                                                   1, s_rnn_b[idx2][:, :-1]).reshape(-1, 1)], axis=1)
        dist2 = np.mean(np.abs(set2[:, 0] - set2[:, 1]))
    except:
        dist2 = dist1


    return (dist1 + dist2) / 2


def ate_experiment(rnn_x,rnn_y,Time,y_pred_t,y_pred1_t,y_pred0_t, s_model,propensity):
    y_pred_t= np.clip(y_pred_t,a_min = 1e-3, a_max= 0.9999)
    y_pred1_t= np.clip(y_pred1_t,a_min = 1e-3, a_max=0.9999)
    y_pred0_t= np.clip(y_pred0_t,a_min = 1e-3, a_max=0.9999)
    s_model= np.clip(s_model,a_min = 1e-3, a_max=0.9999)

    #############################################
    #IPW
    weight1 = rnn_x[Time == 0,0,0] / np.clip(propensity[Time == 0],a_min = 0.1, a_max=0.9) * sum(rnn_x[Time == 0,0,0])/len(y_pred_t)
    weight0 = (1-rnn_x[Time == 0,0,0]) / np.clip(1-propensity[Time == 0],a_min = 0.1,a_max=0.9) * (len(y_pred_t)-sum(rnn_x[Time == 0,0,0]))/len(y_pred_t)

    cf_durv_IPW = []
    hr_durv_IPW = []

    y_pred_t_1_adj = s_model * weight1.reshape(-1,1)
    y_pred_t_0_adj = s_model * weight0.reshape(-1,1)

    h1_adj = y_pred_t * weight1.reshape(-1, 1)
    h0_adj = y_pred_t * weight0.reshape(-1, 1)

    for i in range(y_pred_t.shape[1]):
        cf_durv_IPW.append( np.mean(y_pred_t_0_adj[:,i]) - np.mean(y_pred_t_1_adj[:,i]))
        hr_durv_IPW.append(np.mean(h0_adj[:,i]) / np.mean(h1_adj[:,i]))

    cf_durv_IPW = np.array(cf_durv_IPW)
    hr_durv_IPW = np.array(hr_durv_IPW)


    #############################################
    #TMLE
    propensity = propensity.reshape(-1,1)

    y_pred_t_tmle_1 = y_pred1_t.copy()
    y_pred_t_tmle_0 = y_pred0_t.copy()

    weight1 = rnn_x[Time == 0,0,0].reshape(-1,1)
    weight0 = (1-rnn_x[Time == 0,0,0].reshape(-1,1))
    H1 = weight1 / np.clip(propensity[Time == 0],a_min = 0.1, a_max=1)
    H0 = weight0 / np.clip(1-propensity[Time == 0],a_min = 0.1, a_max=1)


    deltas = []
    for i in range(y_pred_t.shape[1]):
        Y = rnn_y[Time == 0][:, i].reshape(-1,1)
        Y[-1] = 0
        Y[-2] = 1
        Ypred = np.log(y_pred_t[:,i]/(1 - y_pred_t[:,i] )).reshape(-1,1)


        df = pd.DataFrame(np.concatenate([H1,H0, Ypred, Y],1), columns= ['h1','h0','q0','y'])
        reg = smf.glm('y ~ -1 + h1 + h0', data=df, offset= df['q0']).fit()

        deltas.append(np.array(reg.bse[0:2]))

    for i in range(y_pred_t.shape[1]):
        Q1W_logis = np.log(y_pred1_t[:, i] / (1 - y_pred1_t[:, i])).reshape(-1, 1) + deltas[i][0] / np.clip(propensity[Time == 0],a_min = 0.01, a_max=1)
        Q0W_logis = np.log(y_pred0_t[:, i] / (1 - y_pred0_t[:, i])).reshape(-1, 1) + deltas[i][1] / np.clip(1-propensity[Time == 0],a_min = 0.01, a_max=1)

        copy1 = y_pred_t_tmle_1[:,i].copy()
        copy0 = y_pred_t_tmle_0[:,i].copy()

        y_pred_t_tmle_1[:,i] = (np.exp(Q1W_logis)/ (1+np.exp(Q1W_logis) )).reshape(-1)
        y_pred_t_tmle_0[:,i] = (np.exp(Q0W_logis)/(1+np.exp(Q0W_logis))).reshape(-1)
        y_pred_t_tmle_1[np.isnan(y_pred_t_tmle_1[:, i]), i] = copy1[np.isnan(y_pred_t_tmle_1[:, i])]
        y_pred_t_tmle_0[np.isnan(y_pred_t_tmle_0[:, i]), i] = copy0[np.isnan(y_pred_t_tmle_0[:, i])]

    hr_durv_tmle = np.mean(np.cumprod(y_pred_t_tmle_0, 1)* weight1 ,0) / np.mean(np.cumprod(y_pred_t_tmle_1, 1)* weight0 ,0)

    y_pred_t_tmle_1 = np.cumprod(y_pred_t_tmle_1, 1)
    y_pred_t_tmle_0 = np.cumprod(y_pred_t_tmle_0, 1)
    cf_durv_tmle =  np.mean(np.cumprod(y_pred_t_tmle_0, 1)* weight1 - np.cumprod(y_pred_t_tmle_1, 1)* weight0 ,0)

    #############################################
    #############################################

    # sns.set(style="whitegrid", font_scale=1)
    # fig, ax = plt.subplots(figsize=(7, 7))
    # ax.plot(range(max_time), np.mean(cf_true[:, 0:max_time], 0), color="#8c8c8c", label="True")
    # ax.plot(range(max_time), cf_durv_IPW, '--', color="#f0aeb4", label="IPW")
    # ax.plot(range(max_time), cf_durv_tmle, '.', color="#f0aeb4", label="TMLE")
    # ax.plot(range(max_time), np.mean(cf_durv, 0), color='#8DBFC5', alpha=0.9, label="CDSM")
    # ax.plot(range(max_time), np.mean(cf_cdsm, 0), '+', color='#8DBFC5', alpha=1, label="CDSM (NC)")
    # #ax.set_xticklabels(np.arange(-8, max_time * 3, 8))
    # ax.set_xlabel("Time", fontsize=11, fontweight='bold')
    # ax.set_ylabel("ATE (Difference in Survival Probability)", fontsize=11, fontweight='bold')
    # plt.legend()
    # plt.savefig("plots/ate_causal.png", bbox_inches='tight', pad_inches=0.5, dpi=500)
    # plt.show()

    #############################################
    #############################################


    return cf_durv_IPW, cf_durv_tmle, hr_durv_tmle, hr_durv_IPW


def get_rmse_bias(s_true, cf_true, hr_true, s_model, cf_model, hr_model,max_time = 30):
        epsilon = 1e-3
        bias_s = (np.abs(np.mean(s_model- s_true[:,:max_time],0))+epsilon) / (np.mean(s_true[:,:max_time],0)+epsilon)
        bias_cf = (np.abs(np.mean(cf_model- cf_true[:,:max_time],0))+epsilon) / (np.mean(cf_true[:,:max_time],0)+epsilon)
        bias_cf[0] = 0
        bias_hr = (np.abs(np.mean(hr_model- hr_true[:,:max_time],0))+epsilon) / (np.mean(hr_true[:,:max_time],0)+epsilon)


        rmse_s = np.sqrt(np.mean((s_model - s_true[:,:max_time] ) ** 2, 0))
        rmse_cf = np.sqrt(np.mean((cf_model - cf_true[:,:max_time] ) ** 2, 0))
        rmse_hr = np.sqrt(np.mean((hr_model - hr_true[:,:max_time]) ** 2, 0))

        return bias_s, bias_cf, bias_hr, rmse_s, rmse_cf, rmse_hr


def subgroup_analyis(s_true, cf_true,hr_true, s_model, cf_model,hr_model, y_pred_std, cf_std_1, max_time = 30):
        try:

            cf_l = cf_model - 1960 * cf_std_1 / np.sqrt(len(y_pred_std))
            cf_u = cf_model + 1960 * cf_std_1 / np.sqrt(len(y_pred_std))
            s_l = s_model - 1960 * y_pred_std / np.sqrt(len(y_pred_std))
            s_u = s_model + 1960 * y_pred_std / np.sqrt(len(y_pred_std))

            coverage_cf_subB = []
            coverage_sf_subB = []
            for subgroup in [1, 5,10,25,50,len(s_true)]:
                coverage_cf_sub = []
                coverage_sf_sub = []
                size = int(len(s_true) / subgroup)
                for i in range(size):
                    idx = np.random.choice(range(len(y_pred_std)), subgroup, replace= False)
                    coverage_cf_sub_tp = (np.mean(cf_true[idx, :max_time], 0) >= np.mean(cf_l[idx], 0)) & (
                                np.mean(cf_true[idx, :max_time], 0) <= np.mean(cf_u[idx], 0))
                    coverage_sf_sub_tp = (np.mean(s_true[idx, :max_time], 0) >= np.mean(s_l[idx], 0)) & (
                                np.mean(s_true[idx, :max_time], 0) <= np.mean(s_u[idx], 0))
                    coverage_cf_sub.append(coverage_cf_sub_tp.reshape(-1,1))
                    coverage_sf_sub.append(coverage_sf_sub_tp.reshape(-1,1))
                coverage_sf_sub = np.sum(np.concatenate(coverage_sf_sub,1),1)/size
                coverage_cf_sub = np.sum(np.concatenate(coverage_cf_sub,1),1)/size
                coverage_cf_subB.append( np.append(subgroup, coverage_cf_sub))
                coverage_sf_subB.append( np.append(subgroup, coverage_sf_sub))
        except:
            coverage_cf_subB = []
            coverage_sf_subB = []
            cf_l = []
            cf_u = []
            s_l = []
            s_u = []

        try:
            rmse_cf_subB = []
            rmse_sf_subB = []
            rmse_hr_subB = []
            bias_cf_subB = []
            bias_sf_subB = []
            bias_hr_subB = []
            for subgroup in [1, 5,10,25,50,len(s_true)]:
                rmse_c_sub = []
                rmse_s_sub = []
                rmse_hr_sub = []
                bias_c_sub = []
                bias_s_sub = []
                bias_hr_sub = []

                size = int(len(s_true) / subgroup)
                for i in range(size):
                    idx = np.random.choice(range(len(cf_model)), subgroup, replace= False)
                    bias_s, bias_cf, bias_hr, rmse_s, rmse_cf, rmse_hr = get_rmse_bias(s_true[idx], cf_true[idx],
                                                                                       hr_true[idx], s_model[idx],
                                                                                       cf_model[idx], hr_model[idx],max_time = max_time)
                    rmse_c_sub.append(rmse_cf.reshape(-1,1))
                    rmse_s_sub.append(rmse_s.reshape(-1,1))
                    rmse_hr_sub.append(rmse_hr.reshape(-1,1))
                    bias_c_sub.append(bias_cf.reshape(-1, 1))
                    bias_s_sub.append(bias_s.reshape(-1, 1))
                    bias_hr_sub.append(bias_hr.reshape(-1, 1))

                rmse_c = np.mean(np.concatenate(rmse_c_sub,1),1)
                rmse_s = np.mean(np.concatenate(rmse_s_sub,1),1)
                rmse_hr = np.mean(np.concatenate(rmse_hr_sub,1),1)
                bias_c = np.mean(np.concatenate(bias_c_sub,1),1)
                bias_s = np.mean(np.concatenate(bias_s_sub,1),1)
                bias_hr = np.mean(np.concatenate(bias_hr_sub,1),1)

                rmse_cf_subB.append( np.append(subgroup, rmse_c))
                rmse_sf_subB.append( np.append(subgroup, rmse_s))
                rmse_hr_subB.append( np.append(subgroup, rmse_hr))
                bias_cf_subB.append( np.append(subgroup, bias_c))
                bias_sf_subB.append( np.append(subgroup, bias_s))
                bias_hr_subB.append( np.append(subgroup, bias_hr))

        except:
            rmse_cf_subB = []
            rmse_sf_subB = []
            rmse_hr_subB = []
            bias_cf_subB = []
            bias_sf_subB = []
            bias_hr_subB = []
        return coverage_sf_subB, coverage_cf_subB, \
               rmse_cf_subB, rmse_sf_subB, \
               bias_cf_subB,bias_sf_subB, \
               rmse_hr_subB, bias_hr_subB, cf_l, cf_u, s_l, s_u


def compose_coverage(coverage_sf_subB, coverage_cf_subB, rmse_cf_subB, rmse_sf_subB, bias_cf_subB,bias_sf_subB,
                         rmse_hr_subB, bias_hr_subB):
        try:
            coverage_subgroup_sf = pd.DataFrame(coverage_sf_subB).loc[:, 1:].T
            coverage_subgroup_sf.columns = (pd.DataFrame(coverage_sf_subB).loc[:, 0]).astype('int32')
            coverage_subgroup_sf['group'] = 'coverage survival curve'

            coverage_subgroup_cf = pd.DataFrame(coverage_cf_subB).loc[:, 1:].T
            coverage_subgroup_cf.columns = (pd.DataFrame(coverage_cf_subB).loc[:, 0]).astype('int32')
            coverage_subgroup_cf['group'] = 'coverage causal effect'

        except:
            pass

        subgroup_rmse_hr = pd.DataFrame(rmse_hr_subB).loc[:, 1:].T
        subgroup_rmse_hr.columns = (pd.DataFrame(rmse_hr_subB).loc[:, 0]).astype('int32')
        subgroup_rmse_hr['group'] = 'rmse hazard ratio'


        subgroup_rmse_cf = pd.DataFrame(rmse_cf_subB).loc[:, 1:].T
        subgroup_rmse_cf.columns = (pd.DataFrame(rmse_cf_subB).loc[:, 0]).astype('int32')
        subgroup_rmse_cf['group'] = 'rmse causal effect'


        subgroup_rmse_sf = pd.DataFrame(rmse_sf_subB).loc[:, 1:].T
        subgroup_rmse_sf.columns = (pd.DataFrame(rmse_sf_subB).loc[:, 0]).astype('int32')
        subgroup_rmse_sf['group'] = 'rmse survival curve'

        subgroup_bias_hr = pd.DataFrame(bias_hr_subB).loc[:, 1:].T
        subgroup_bias_hr.columns = (pd.DataFrame(bias_hr_subB).loc[:, 0]).astype('int32')
        subgroup_bias_hr['group'] = 'bias hazard ratio'

        subgroup_bias_cf = pd.DataFrame(bias_cf_subB).loc[:, 1:].T
        subgroup_bias_cf.columns = (pd.DataFrame(bias_cf_subB).loc[:, 0]).astype('int32')
        subgroup_bias_cf['group'] = 'bias causal effect'

        subgroup_bias_sf = pd.DataFrame(bias_sf_subB).loc[:, 1:].T
        subgroup_bias_sf.columns = (pd.DataFrame(bias_sf_subB).loc[:, 0]).astype('int32')
        subgroup_bias_sf['group'] = 'bias survival curve'

        try:
            subgroup_df = pd.concat([coverage_subgroup_sf, coverage_subgroup_cf, subgroup_rmse_cf, subgroup_rmse_sf,
                                     subgroup_bias_cf, subgroup_bias_sf, subgroup_bias_hr, subgroup_rmse_hr]).reset_index()
        except:
            subgroup_df = pd.concat([subgroup_rmse_cf, subgroup_rmse_sf,
                                     subgroup_bias_cf, subgroup_bias_sf,
                                     subgroup_bias_hr, subgroup_rmse_hr]).reset_index()
        return  subgroup_df


def get_evaluation_true(data, trueSurv_wrapper, propensity,
                        y_pred_t, y_pred_std, y_pred1_t, y_pred0_t, cf_std_1, algo = 'CDSM', max_time = 30 ):

    rnn_x, rnn_m, rnn_s, rnn_y, Time = data
    trueSurv, trueSurv_1, trueSurv_0 = trueSurv_wrapper

    if np.sign(np.mean(trueSurv_1-trueSurv_0)) != np.sign(np.mean(y_pred1_t-y_pred0_t)):
        temp = y_pred1_t.copy()
        y_pred1_t = y_pred0_t.copy()
        y_pred0_t = temp.copy()

    s_true = np.cumprod(trueSurv,1)
    cf_true = np.cumprod(trueSurv_0,1) - np.cumprod(trueSurv_1,1)
    hr_true = trueSurv_0 / trueSurv_1
    s_model = np.cumprod(y_pred_t, 1)
    cf_model = np.cumprod(y_pred0_t, 1) - np.cumprod(y_pred1_t, 1)
    hr_model = np.clip(y_pred0_t, 0.001, 1) / np.clip(y_pred1_t, 0.001, 1)

    cf_durv_IPW, cf_durv_tmle, hr_durv_tmle, hr_durv_IPW = \
        ate_experiment(rnn_x,rnn_y,Time,y_pred_t,y_pred1_t,y_pred0_t, s_model,propensity)


    AUROC = custom_auc(s_model, rnn_y[Time == 0, 0:max_time], plot=False)
    concordance = get_concordance(s_model, rnn_y[Time == 0, 0:max_time],rnn_y[Time == 0, max_time:2 * max_time])

    res = minimize(lambda x: get_distance(s_model, rnn_y, Time, max_time=max_time, threshold=x), 0.9,
                   method='nelder-mead', options={'xatol': 1e-7, 'disp': True})
    avg_dist = get_distance(s_model, rnn_y, Time, threshold=res.x, max_time=max_time)
    dist_var = np.var(
        np.apply_along_axis(lambda x: get_distance(s_model, rnn_y, Time, threshold=x, max_time=max_time), 0,
                            np.arange(0.9, 1, 0.01).reshape(1, -1)))


    coverage_sf_subB, coverage_cf_subB, \
    rmse_cf_subB, rmse_sf_subB, \
    bias_cf_subB, bias_sf_subB, \
    rmse_hr_subB, bias_hr_subB, cf_l, cf_u, s_l, s_u =  \
        subgroup_analyis(s_true, cf_true,hr_true, s_model, cf_model,hr_model, y_pred_std, cf_std_1)

    subgroup_df = compose_coverage(coverage_sf_subB, coverage_cf_subB, rmse_cf_subB, rmse_sf_subB, bias_cf_subB,
                                   bias_sf_subB, rmse_hr_subB, bias_hr_subB)

    subgroup_df_sel = subgroup_df[(subgroup_df['index']>4) & (subgroup_df['index']<20)]
    subgroup_avg = subgroup_df_sel.groupby(['group']).mean()
    subgroup_std = subgroup_df_sel.groupby(['group']).std()
    subgroup_std = subgroup_std.loc[['bias causal effect', 'bias hazard ratio', 'bias survival curve']]
    subgroup_std.index = ['bias std causal effect', 'bias std hazard ratio', 'bias std survival curve']

    metrics = pd.DataFrame([AUROC, concordance, avg_dist, dist_var],index = ['AUROC','Concordance','Distance','Distance Std'])
    metrics = pd.concat([metrics, subgroup_avg[max(subgroup_avg.columns[2:])], subgroup_std[max(subgroup_avg.columns[2:])]])
    metrics = metrics.T

    IPW_bias = abs(cf_durv_IPW - np.mean(cf_true[:,0:max_time]))/np.mean(cf_true[:,0:max_time])
    TMLE_bias = abs(cf_durv_tmle - np.mean(cf_true[:,0:max_time]))/np.mean(cf_true[:,0:max_time])

    metrics['bias cf (ipw)'] = np.mean(IPW_bias[5:20])
    metrics['bias cf (tmle)'] = np.mean(TMLE_bias[5:20])

    IPW_bias = abs(hr_durv_IPW - np.mean(hr_true[:, 0:max_time])) / np.mean(hr_true[:, 0:max_time])
    TMLE_bias = abs(hr_durv_tmle - np.mean(hr_true[:, 0:max_time])) / np.mean(hr_true[:, 0:max_time])

    metrics['bias hr (ipw)'] = np.mean(IPW_bias[5:20])
    metrics['bias hr (tmle)'] = np.mean(TMLE_bias[5:20])

    if 'coverage survival curve' not in metrics.columns:
        metrics['coverage survival curve'] = 0
        metrics['coverage causal effect'] = 0

    metrics['algorithm'] = algo
    subgroup_df['algorithm'] = algo

    return metrics,subgroup_df, s_model, cf_model, hr_model ,s_true, cf_true, hr_true, cf_l, cf_u, s_l, s_u



