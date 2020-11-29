import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from tqdm import tqdm
import tensorflow as tf

from utilss.data_handler import LDataSimu, build_data_surv_rnn, arr_to_xmd
from utilss.model import create_model, get_counterfactuals, benchmark_algorithms
from utilss.Evaluations import custom_auc, get_concordance
from joblib import Parallel, delayed
from scipy.optimize import minimize


AFdata = pd.read_csv('dataset/af_vka_vs_noac.csv')
AFdata = AFdata.sort_values(by=['patnumber', 'follow_up']).reset_index(drop = True)
Colnames = ['patnumber', 'imd2015',
       'age', 'appendage_occlusion', 'cardioversion', 'ablation', 'paroxysmal',
       'persistent_chronic', 'timefromdiagnosis', 'esrf_count',
       'mi_count', 'chf_count', 'pvd_count', 'cvd_count', 'dem_count',
       'copd_count', 'rhe_count', 'giu_count', 'liv_m_count', 'dtm_u_count',
       'dtm_c_count', 'hp_count', 'ckd_count', 'can_count', 'can_met_count',
       'liv_s_count', 'aids_count', 'charlson_index_2y', 'female',
       'hyp_count', 'stroke_count', 'chadvasc_index_2y', 'bleed_count',
       'l_inr_count', 'alch_count', 'hasbled_index_2y',
       'Pacemaker system procedures_count', 'Antiarrythmics_count',
       'Antidiabetics_count', 'Antihypertensives_count', 'Aspirin_count',
       'Betablockers_count', 'CP450_Inhibitors_count',
       'Parenteral Anticoagulants_count', 'Proton Pump Inhibitors_count',
       'Rifampicin_count', 'Selected Anticonvulsants_count',
       'Selective Serotonin Re-uptake Inhibitors_count', 'Statins_count',
       'NSAID_count', 'Antiplatelets_count']

print(len(np.unique(AFdata[((AFdata.naive_vka==1))]['patnumber'])))


X = AFdata.copy()
X = X[Colnames]
np.mean(X,0) == 0

##########################################################
# preprocess the data
def AF_model(df, model_name ='dSurv_af.pkl', model_path = 'model_result_af_perprotocal', load = False, build_data = True, layers = 30):
    train_idx, validate_idx, test_idx = np.split(df.index, [int(.9 * len(df.index)), int(.99 * len(df.index))])
    train = df.iloc[train_idx]
    val = df.iloc[validate_idx]
    test = df.iloc[test_idx]

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
        return train, one_X

    train_scaled, train_stat = preprocess_data(train)
    val_scaled,val_stat = preprocess_data(val)
    test_scaled,val_stat = preprocess_data(test)

    def build_data_rnn(train, score = None, history_itvl=14, prediction_itvl=1, max_time=20):
        observation = train.patnumber
        one_X = train.groupby(train.patnumber)['T'].max()
        event = train.groupby(train.patnumber)['Y'].max()
        train.drop(['T', 'Y', 'patnumber'], axis=1, inplace=True)
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
                utility_obs[:, 0:max_engine_time + 1, :] = score[observation == i, :]
            else:
                rnn_utility = np.empty((0, max_time, 3), dtype=np.float32)
                utility_obs = np.zeros((1, max_time, 3), dtype=np.float32)

            if f[i]:
                yobs[0, 0:max_engine_time] = 1.0
                try:
                    yobs[1, max_engine_time] = 1.0
                except:
                    pass
            else:
                yobs[0, 0:max_engine_time + 1] = 1.0

            start = max(0, max_engine_time - prediction_itvl)
            end = max(max_engine_time, 2)
            step = 1
            count = 0
            for j in np.arange(end, start, -step):
                covariate_x = train_x[observation == i, :]
                # covariate_x[0,:] = covariate_x[1,:]
                xtemp = np.ones((1, history_itvl, x_dim), dtype=np.float32) * -1
                x_end = min(j + 1, max_engine_time)
                x_start = max(x_end - history_itvl, 0)
                x_end_t = covariate_x[x_start:x_end, :].reshape((1, -1, x_dim)).shape
                xtemp[0, 0:x_end_t[1], :] = covariate_x[x_start:x_end, :].reshape((1, -1, x_dim))

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

                x_temp, m_temp, s_temp = arr_to_xmd(xtemp)

                rnn_y = np.concatenate((rnn_y, ytemp))
                rnn_x = np.concatenate((rnn_x, x_temp))
                rnn_m = np.concatenate((rnn_m, m_temp))
                rnn_s = np.concatenate((rnn_s, s_temp))
                rnn_utility = np.concatenate((rnn_utility, utility_obs))
                ID = np.concatenate((ID, np.array(i).reshape(1)))  # np.array(i).reshape(1)  #
                Time = np.concatenate((Time, np.array(count).reshape(1)))  # np.array(j).reshape(1)  #
                Event = np.concatenate((Event, _event))  # _event  #
                count = count + 1

            return rnn_y, rnn_x, rnn_m, rnn_s, ID, Time, Event, rnn_utility

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

    if build_data:
        results = build_data_rnn(train_scaled, score=None)
        np.save('dataset/rnn_train_af',{'res':results})

        results = build_data_rnn(val_scaled, score=None)
        np.save('dataset/rnn_val_af',{'res':results})

        results = build_data_rnn(test_scaled, score=None)
        np.save('dataset/rnn_test_af',{'res':results})


    #Load data
    res = np.load('dataset/rnn_train_af.npy', allow_pickle=True)
    rnn_x, rnn_m, rnn_s, rnn_y, ID, Time, Event, rnn_utility = res.tolist()['res']
    rnn_x[np.isnan(rnn_x)] = 0

    res = np.load('dataset/rnn_val_af.npy', allow_pickle=True)
    test = res.tolist()['res']
    rnn_x_val, rnn_m_val, rnn_s_val, rnn_y_val, ID_val, Time_val, Event_val, rnn_utility_val = res.tolist()['res']
    rnn_x_val[np.isnan(rnn_x_val)] = 0

    res = np.load('dataset/rnn_test_af.npy', allow_pickle=True)
    rnn_x_test, rnn_m_test, rnn_s_test, rnn_y_test,ID_test, Time_test, Event_test, rnn_utility_test = res.tolist()['res']
    rnn_x_test[np.isnan(rnn_x_test)] = 0



    # Train the model
    history_itvl = 14
    max_time = 20
    data = [rnn_x, rnn_m, rnn_s, rnn_y, Time]
    val_data = [rnn_x_val, rnn_m_val, rnn_s_val, rnn_y_val,Time_val]
    test_data = [rnn_x_test, rnn_m_test, rnn_s_test, rnn_y_test, Time_test]


    pred_rnn_x = np.concatenate([rnn_x,rnn_x_val,rnn_x_test])
    pred_rnn_m = np.concatenate([rnn_m,rnn_m_val,rnn_m_test])
    pred_rnn_s = np.concatenate([rnn_s,rnn_s_val,rnn_s_test])
    pred_rnn_y = np.concatenate([rnn_y,rnn_y_val,rnn_y_test])
    pred_time = np.concatenate([Time,Time_val, Time_test])
    pred_data = [pred_rnn_x, pred_rnn_m, pred_rnn_s, pred_rnn_y, pred_time]

    # KM model
    modelKM = benchmark_algorithms(rnn_x.shape[2], max_time, history_itvl, data, val_data, one_X=train_stat,
                                   lstm_window=14, model="KM")
    modelKM_result = get_counterfactuals(modelKM, pred_data, t=0, draw=1, type="KM")
    y_pred_t, y_pred_std, y_pred1_t, y_pred0_t, cf_std_1, time_pt, \
    y_pred_t_test, y_pred_std_test, y_pred1_t_test, y_pred0_t_test, cf_std_1_test, time_pt_test = modelKM_result
    km_idx = y_pred_t.survival_function_.index.values
    s_km = y_pred_t.survival_function_.KM_estimate
    cf_km = y_pred1_t.survival_function_.KM_estimate - y_pred0_t.survival_function_.KM_estimate


    #Load model
    if load:
        res = np.load('saved_models/'+model_path+'.npy', allow_pickle=True).tolist()
        modelDsurv_result = np.array(res['modelDsurv_result'])
    else:
        # Model fitting lstm model with censor loss
        modelDsurv,model_p, history_dict = create_model(rnn_x.shape[2], max_time, history_itvl, data, val_data, lstm_window= 6,
                                                alpha=1.5,beta= 1, gamma=0.5, load=load, verbose=0, model_name=model_name,
                                                batch_size=2056, layers=30)
        modelDsurv_result = get_counterfactuals(modelDsurv, pred_data, t=0, draw=10, test_data=test_data)

        propensity_cdsm = model_p([pred_rnn_x, pred_rnn_m])[:,0]


        np.save('saved_models/' + model_path + '.npy', {'modelDsurv_result': modelDsurv_result})

    y_pred_t, y_pred_std, y_pred1_t, y_pred0_t, cf_std_1, _, \
    y_pred_t_test, y_pred_std_test, y_pred1_t_test, y_pred0_t_test, cf_std_1_test, _ = modelDsurv_result
    s_durv = np.cumprod(y_pred_t, 1)
    cf_durv = np.cumprod(y_pred1_t, 1) - np.cumprod(y_pred0_t, 1)
    hr_durv = np.clip(y_pred0_t, 0.001, 1) / np.clip(y_pred1_t, 0.001, 1)



    #plots
    sns.set(style="whitegrid", font_scale=1)
    fig, ax = plt.subplots(figsize=(7, 7))
    ax.plot(range(max_time), s_km.iloc[0:max_time], color="#8c8c8c", label="KM")
    ax.plot(range(max_time), np.mean(s_durv, 0), color="#8DBFC5", label="CDSM")
    ax.fill_between(range(max_time), np.quantile(s_durv, q = 0.25, axis = 0), np.quantile(s_durv, q = 0.75, axis = 0),
                    facecolor='#8DBFC5', alpha=0.4)
    ax.fill_between(range(max_time), np.quantile(s_durv, q = 0.05, axis = 0), np.quantile(s_durv, q = 0.95, axis = 0),
                    facecolor='#8DBFC5', alpha=0.2)
    ax.set_xticklabels(np.arange(-8,max_time*3,8))
    ax.set_xlabel("Months", fontsize=11, fontweight='bold')
    ax.set_ylabel("Survival Probability", fontsize=11, fontweight='bold')
    plt.legend()
    plt.savefig("plots/"+model_path+"_survival.png", bbox_inches='tight', pad_inches=0.5, dpi=500)
    plt.show()

    sns.set(style="whitegrid", font_scale=1)
    fig, ax = plt.subplots(figsize=(7, 7))
    ax.plot(range(max_time), cf_km.iloc[0:max_time], '--', color="#8c8c8c", label="KM")
    ax.plot(range(max_time), np.mean(cf_durv, 0), color='#8DBFC5', alpha=0.9, label="D-Surv")
    ax.fill_between(range(max_time), np.quantile(cf_durv, q=0.25, axis=0), np.quantile(cf_durv, q=0.75, axis=0),
                    facecolor='#8DBFC5', alpha=0.4)
    ax.fill_between(range(max_time), np.quantile(cf_durv, q=0.05, axis=0), np.quantile(cf_durv, q=0.95, axis=0),
                    facecolor='#8DBFC5', alpha=0.2)
    ax.set_xticklabels(np.arange(-8, max_time * 3, 8))
    ax.set_xlabel("Months", fontsize=11, fontweight='bold')
    ax.set_ylabel("Difference in Survival Probability", fontsize=11, fontweight='bold')
    plt.legend()
    plt.savefig("plots/" + model_path + "_causal.png", bbox_inches='tight', pad_inches=0.5, dpi=500)
    plt.show()


    sns.set(style="whitegrid", font_scale=1)
    fig, ax = plt.subplots(figsize=(7, 7))
    ax.plot(range(max_time), np.mean(hr_durv, 0), color="#8DBFC5", label="D-Surv")
    ax.fill_between(range(max_time), np.quantile(hr_durv, q=0.25, axis=0), np.quantile(hr_durv, q=0.75, axis=0),
                    facecolor='#8DBFC5', alpha=0.5)
    ax.fill_between(range(max_time), np.quantile(hr_durv, q=0.05, axis=0), np.quantile(hr_durv, q=0.95, axis=0),
                    facecolor='#8DBFC5', alpha=0.2)
    ax.set_xticklabels(np.arange(-8,max_time*3,8))
    ax.set_xlabel("Months", fontsize=11, fontweight='bold')
    ax.set_ylabel("Hazard Ratio*", fontsize=11, fontweight='bold')
    plt.legend()
    plt.savefig("plots/" + model_path + "_hr.png", bbox_inches='tight', pad_inches=0.5, dpi=500)
    plt.show()



    def get_distance(s_rnn_b,pred_rnn_y,pred_time,threshold, max_time):
        event = np.sum(pred_rnn_y[pred_time == 0, max_time:2 * max_time], axis = 1)
        censor = np.sum(pred_rnn_y[pred_time == 0, 0: max_time], axis = 1)
        idx1 = np.where(event>=1)
        idx2 = np.where((event==0) & (censor!=max_time))
        set1 = np.concatenate([np.where(np.diff( pred_rnn_y[pred_time == 0, 0:max_time][idx1] ) == -1)[1].reshape(-1,1) ,
                               np.apply_along_axis(lambda x: np.min(np.where(x  <= threshold) if len(np.where(x <=threshold)[0]) > 0 else max_time),1,
                                                   s_rnn_b[idx1][:,:-1]).reshape(-1,1)], axis=1)

        set2 = np.concatenate([np.where(np.diff(pred_rnn_y[pred_time == 0, 0:max_time][idx2]) == -1)[1].reshape(-1, 1),
                               np.apply_along_axis(lambda x: np.min( np.where(x <= threshold) if len(np.where(x <= threshold)[0]) > 0 else max_time),
                                                   1, s_rnn_b[idx2][:, :-1]).reshape(-1, 1)], axis=1)

        dist1 = np.mean(np.abs( set1[:,0] - set1[:,1]) )
        dist2 = np.mean(np.abs( set2[:,0] - set2[:,1]) )
        return (dist1+dist2)/2
    def get_evaluation(model_result, pred_rnn_y, rnn_y_test,pred_time, Time_test, max_time):
        # Standard lstm model with binary loss
        y_pred_t, y_pred_std, y_pred1_t, y_pred0_t, cf_std_1, _,\
        y_pred_t_test, y_pred_std_test, y_pred1_t_test, y_pred0_t_test, cf_std_1_test, _ = model_result
        s_rnn_b = np.cumprod(y_pred_t, 1)
        cf_rnn_b = np.cumprod(y_pred0_t, 1) - np.cumprod(y_pred1_t, 1)
        s_rnn_b_test = np.cumprod(y_pred_t_test, 1)

        AUROC_b = custom_auc(s_rnn_b, pred_rnn_y[pred_time == 0, 0:max_time], plot=False)
        concordance_b = get_concordance(s_rnn_b, pred_rnn_y[pred_time == 0, 0:max_time], pred_rnn_y[pred_time == 0, max_time:2 * max_time])
        AUROC_b_test = custom_auc(np.cumprod(y_pred_t_test, 1), rnn_y_test[Time_test == 0, 0:max_time], plot=False)
        concordance_b_test = get_concordance(np.cumprod(y_pred_t_test, 1), rnn_y_test[Time_test == 0, 0:max_time], rnn_y_test[Time_test == 0, max_time:2 * max_time])

        res = minimize(lambda x: get_distance(s_rnn_b,pred_rnn_y,pred_time,max_time=max_time, threshold=x), 0.9, method='nelder-mead', options={'xatol': 1e-7, 'disp': True})
        avg_dist =  get_distance(s_rnn_b,pred_rnn_y,pred_time, threshold=res.x, max_time = max_time)
        avg_dist_test =  get_distance(np.cumprod(y_pred_t_test, 1),rnn_y_test,Time_test, threshold=res.x,max_time=max_time)

        dist_var = np.var(np.apply_along_axis(lambda x:get_distance(s_rnn_b,pred_rnn_y,pred_time, threshold= x,max_time=max_time), 0, np.arange(0.9,1,0.01).reshape(1,-1) ))
        dist_var_test =  np.var(np.apply_along_axis(lambda x:get_distance(s_rnn_b_test,rnn_y_test,Time_test, threshold= x,max_time=max_time), 0, np.arange(0.9,1,0.01).reshape(1,-1) ))

        print(AUROC_b,concordance_b,AUROC_b_test,concordance_b_test, avg_dist , avg_dist_test, dist_var, dist_var_test)
        return AUROC_b,concordance_b,AUROC_b_test,concordance_b_test, avg_dist , avg_dist_test, dist_var, dist_var_test, s_rnn_b, cf_rnn_b


    AUROC,concordance,AUROC_test,concordance_test,avg_dist,avg_dist_test, dist_var, \
    dist_var_test, s_durv, cf_durv = get_evaluation(modelDsurv_result, pred_rnn_y, rnn_y_test,pred_time, Time_test, max_time)

    _, y_pred_std, _, _, cf_std_1, _, _, _, _, _, _, _ = modelDsurv_result
    cf_l_durv = cf_durv - 1.96*cf_std_1/np.sqrt(len(cf_std_1))
    cf_u_durv = cf_durv + 1.96*cf_std_1/np.sqrt(len(cf_std_1))
    # s_l_durv = s_durv - 1.96*y_pred_std/np.sqrt(len(cf_std_1))
    # s_u_durv = s_durv + 1.96*y_pred_std/np.sqrt(len(cf_std_1))


    idx_kernal = pd.DataFrame(pred_rnn_x[pred_time==0][:,0,:].reshape(-1,54), columns=Colnames)
    sns.set(style="whitegrid", font_scale=1)
    fig, ax = plt.subplots(figsize=(7, 7))
    sns.kdeplot(np.mean(cf_durv[idx_kernal.patnumber == 1, :],1), alpha = 0.3, label="NOAC", fill = True, color = "#8DBFC5")
    sns.kdeplot(np.mean(cf_durv[idx_kernal.patnumber == 0, :],1), alpha = 0.5, label="VKA", fill = True, color = "#f0aeb4")
    ax.set_xlabel("ITE (Positive values favour NOAC)", fontsize=11, fontweight='bold')
    plt.legend()
    plt.savefig("plots/"+model_path+"_dense.png", bbox_inches='tight', pad_inches=0.5, dpi=500)
    plt.show()



    def gradient_importance(data,pred_time, model, tp = 0, propensity = True):
        rnn_x, rnn_m, rnn_s, rnn_y, _ = data
        if propensity:
            rnn_x = rnn_x[:,:,1:]
            rnn_m = rnn_m[:,:,1:]

        rnn_m0 = rnn_m.copy()
        rnn_m0[rnn_x[:, 0, 0] != 0, :, :] = -1
        rnn_m1 = rnn_m.copy()
        rnn_m1[rnn_x[:, 0, 0] != 1, :, :] = -1

        x = tf.convert_to_tensor( rnn_x[pred_time == tp], name= 'x')
        with tf.GradientTape() as tape:
            tape.watch(x)
            #predictions = model([x, rnn_m[pred_time == tp], rnn_m0[pred_time == tp],
            #                     rnn_m1[pred_time == tp], rnn_s[pred_time == tp]])
            predictions = model([x, rnn_m[pred_time == tp]])
        grads = tape.gradient(predictions, x) #dy/dx
        grads = tf.reduce_mean(grads, axis=1).numpy()[0]
        return grads

    grads = gradient_importance(pred_data, pred_time, model = model_p, tp = 0)
    var_name = Colnames.copy()[1:]
    #var_name[0] = 'NOAC'
    plt_df = pd.DataFrame(var_name)
    plt_df['grads'] = grads
    plt_df.columns = ['x', 'grads']
    plt_df = plt_df.sort_values(['grads'], ascending = False)
    plt_df = plt_df[plt_df.x != 'vka']
    plt_df.x = plt_df.x.str.replace('_count','')
    plt_df.x = plt_df.x.str.replace('_',' ')
    plt_df.x = plt_df.x.str.upper()
    sns.set(style="whitegrid", font_scale=1)
    fig, ax = plt.subplots(figsize=(7, 10))
    sns.barplot(x=plt_df.grads, y=plt_df.x, palette='vlag' )
    ax.set_xlabel("Covariates", fontsize=11, fontweight='bold')
    ax.set_xlabel("Gradients", fontsize=11, fontweight='bold')
    plt.savefig("plots/"+model_path+"_grads.png", bbox_inches='tight', pad_inches=0.5, dpi=500)
    plt.show()


    from scipy.stats import ttest_ind
    obs_negative = pd.DataFrame(pred_rnn_x[pred_time==0][:,0,:][np.mean(cf_durv,1)<0], columns=Colnames)
    obs_positive = pd.DataFrame(pred_rnn_x[pred_time==0][:,0,:][np.mean(cf_durv,1)>0.0001], columns=Colnames)
    stat, p = ttest_ind(obs_negative, obs_positive)
    obs_neg_des = obs_negative.describe().T
    obs_pos_des = obs_positive.describe().T
    obs_diff =  obs_pos_des - obs_neg_des
    obs_result = pd.DataFrame(obs_diff['mean'])
    obs_result['p-value'] = p


    result = pd.DataFrame(np.concatenate([np.mean(cf_durv, 0).reshape(-1, 1),
                                          np.quantile(cf_durv, q=0.25, axis=0).reshape(-1, 1),
                                          np.quantile(cf_durv, q=0.75, axis=0).reshape(-1, 1),
                                          (np.std(cf_durv, 0)/np.sqrt(len(cf_durv)/2)).reshape(-1, 1)], 1),
                          columns=["ATE", "ATE 25%", "ATE 75%", "ATE (SE)"])
    result2 = pd.DataFrame(np.concatenate([np.mean(hr_durv, 0).reshape(-1, 1),
                                          np.quantile(hr_durv, q=0.25, axis=0).reshape(-1, 1),
                                          np.quantile(hr_durv, q=0.75, axis=0).reshape(-1, 1),
                                           (np.std(hr_durv, 0)/np.sqrt(len(hr_durv)/2)).reshape(-1, 1)], 1),
                          columns=["HR", "HR 25%", "HR 75%", "HR (SE)"])
    result = pd.concat([result, result2])
    result = pd.concat([result, pd.DataFrame(
        {AUROC, concordance, AUROC_test, concordance_test, avg_dist, avg_dist_test, dist_var, \
         dist_var_test})])
    result = pd.concat([result, obs_result])
    result.to_csv("results/"+model_path+".csv")
    pd.DataFrame(cf_durv).to_csv("results/cf_durv_"+model_path+".csv")
    pd.DataFrame(cf_l_durv).to_csv("results/cf_l_durv_"+model_path+".csv")
    pd.DataFrame(cf_u_durv).to_csv("results/cf_u_durv_"+model_path+".csv")


    return obs_result, AUROC,concordance,AUROC_test,concordance_test,avg_dist,avg_dist_test, dist_var, \
           dist_var_test, s_durv, cf_durv


Y = pd.DataFrame(np.sum(AFdata[['mb_sw_dc', 'isse_sw_dc', 'death_sw_dc']].copy(), axis=1) >0 ,columns=['Y']) * 1.0
A = pd.DataFrame(AFdata['naive_noac'].values.copy(),columns=['A'])
T = pd.DataFrame(AFdata['follow_up'].values.copy(),columns=['T'])
df = pd.concat([Y,T,A,X],axis=1).reset_index(drop = True)
df = df[~pd.isna(df.Y)].reset_index(drop = True)
len(np.unique(df[((df.A==0) & (df.Y== 1))]['patnumber']))/len(np.unique(df[((df.A==0))]['patnumber']))
len(np.unique(df[((df.A==1) & (df.Y== 1))]['patnumber']))/len(np.unique(df[((df.A==1))]['patnumber']))

obs_result, AUROC,concordance,AUROC_test,concordance_test,avg_dist,avg_dist_test, dist_var, \
           dist_var_test, s_durv, cf_durv = AF_model(df, model_name ='cdsm_af.pkl', model_path = 'model_result_af_perprotocal',
                                                     layers = 30, load = False, build_data = False)




Y = pd.DataFrame(np.sum(AFdata[['mb_sw_dc']].copy(), axis=1) >0 ,columns=['Y']) * 1.0
A = pd.DataFrame(AFdata['naive_noac'].values.copy(),columns=['A'])
T = pd.DataFrame(AFdata['follow_up'].values.copy(),columns=['T'])
df = pd.concat([Y,T,A,X],axis=1).reset_index(drop = True)
len(np.unique(df[(df.Y==1) & (df.A==0)]['patnumber']))
len(np.unique(df[(df.Y==1) & (df.A==1)]['patnumber']))

obs_result_mb, AUROC_mb,concordance_mb,AUROC_test_mb,concordance_test_mb,avg_dist_mb,avg_dist_test_mb, dist_var_mb, \
           dist_var_test_mb, s_durv_mb, cf_durv_mb = AF_model(df, model_name ='dSurv_af_mb.pkl', model_path = 'model_result_af_perprotocal_mb',layers = 26,
                                                              load=True, build_data=False)





Y = pd.DataFrame(np.sum(AFdata[['isse_sw_dc']].copy(), axis=1) >0 ,columns=['Y']) * 1.0
A = pd.DataFrame(AFdata['naive_noac'].values.copy(),columns=['A'])
T = pd.DataFrame(AFdata['follow_up'].values.copy(),columns=['T'])
df = pd.concat([Y,T,A,X],axis=1).reset_index(drop = True)
len(np.unique(df[(df.Y==1) & (df.A==0)]['patnumber']))
len(np.unique(df[(df.Y==1) & (df.A==1)]['patnumber']))

obs_result_isse, AUROC_isse,concordance_isse,AUROC_test_isse,concordance_test_isse,avg_dist_isse,avg_dist_test_isse, dist_var_isse, \
           dist_var_test_isse, s_durv_isse, cf_durv_isse = AF_model(df, model_name ='dSurv_af_isse.pkl', model_path = 'model_result_af_perprotocal_isse',layers = 20,
                                                                    load=True, build_data=False)




Y = pd.DataFrame(np.sum(AFdata[['death_sw_dc']].copy(), axis=1) >0 ,columns=['Y']) * 1.0
A = pd.DataFrame(AFdata['naive_noac'].values.copy(),columns=['A'])
T = pd.DataFrame(AFdata['follow_up'].values.copy(),columns=['T'])
df = pd.concat([Y,T,A,X],axis=1).reset_index(drop = True)
len(np.unique(df[(df.Y==1) & (df.A==0)]['patnumber']))
len(np.unique(df[(df.Y==1) & (df.A==1)]['patnumber']))


obs_result_isse, AUROC_isse,concordance_isse,AUROC_test_isse,concordance_test_isse,avg_dist_isse,avg_dist_test_isse, dist_var_isse, \
           dist_var_test_isse, s_durv_isse, cf_durv_isse = AF_model(df, model_name ='dSurv_af_death.pkl', model_path = 'model_result_af_perprotocal_death',layers = 25,
                                                                    load=True, build_data=False)









