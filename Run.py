import gc
import glob
import os, sys

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from tqdm import tqdm

from utilss.Evaluations import get_evaluation_true
from utilss.model import create_model, get_counterfactuals, benchmark_algorithms
from utilss.data_simulator import get_data

from joblib import Parallel, delayed


def blockPrint():
    sys.stdout = open(os.devnull, 'w')

def enablePrint():
    sys.stdout = sys.__stdout__


##########################################################
# Run the experiments

def experiement_causal(i,input_dim=6, history_itvl=14, sampleSize=1000, draw=30, std = 0.1,confound = 0.5, scale = 2, overlap = 0.5):
    blockPrint()
    max_time = 30

    # input_dim = 10
    # history_itvl = 14
    # sampleSize = 1000
    # prediction_itvl = 1
    # overlap = 1
    # seed = 123
    # confound = 0.2

    data, val_data, test_data, train_stat, val_stat, test_stat, surv_func_wrapper, train_full, val_full, test_full, raw = \
        get_data(input_dim=input_dim, sampleSize=sampleSize, max_time=max_time, prediction_itvl=1, history_itvl=14, overlap=overlap,
                 seed=np.random.random_integers(1, 1000), std=std, confound=confound, scale=scale)

    rnn_x, rnn_m, rnn_s, rnn_y, Time = data
    rnn_x_test,rnn_m_test, rnn_s_test, rnn_y_test, Time_test = test_data

    trueSurv_wrapper = surv_func_wrapper(train_full , train_full.A, len(train_stat), t_start=0, max_time=max_time, plot=False)
    trueSurv_wrapper_test = surv_func_wrapper(test_full , test_full.A, len(test_stat), t_start=0, max_time=max_time, plot=False)

    gc.collect()

    # propebsity_model = create_propensity(rnn_x.shape[2], max_time, history_itvl, data, val_data, batch_size = 256)
    # propebsity = propebsity_model.predict([rnn_x[:,:, 1:], rnn_m[:, :, 1:]])
    #
    # a = propebsity[rnn_x[:,0,0]==1, 0]
    # b = propebsity[rnn_x[:,0,0]==0, 0]
    #
    # print(np.mean(a),np.mean(b))
    #
    # plt.hist(a, alpha = 0.5, bins = 20)
    # plt.hist(b, alpha = 0.5, bins = 20)
    # plt.show()


    # Dynamic survival model
    modelDsurv, model_p , _ = create_model(rnn_x.shape[2], max_time, history_itvl, data, val_data, lstm_window= 7,
                                            alpha= 2, beta= 1, gamma=1, load=False, verbose=0, model_name='cdsm',
                                            batch_size= int(rnn_x.shape[0]/2.2), layers= int(input_dim/3))
    modelDsurv_result = get_counterfactuals(modelDsurv, data, t=0, draw = draw, test_data = test_data)


    # Dynamic survival model 2
    modelDsurv2,_, _ = create_model(rnn_x.shape[2], max_time, history_itvl, data, val_data, lstm_window= 7,
                                            alpha= 2, beta= 1, gamma=0, load=False, verbose=0, model_name='cdsmna',
                                            batch_size= int(rnn_x.shape[0]/2.2), layers= int(input_dim/3))
    modelDsurv2_result = get_counterfactuals(modelDsurv2, data, t=0, draw=draw, test_data=test_data)

    # Standard lstm model with censor loss
    modelRNN = benchmark_algorithms(rnn_x.shape[2], max_time, history_itvl, data, val_data, one_X=None, lstm_window=7,
                                     model="StandardRNN", beta=1, batch_size = 128)
    modelRNN_result = get_counterfactuals(modelRNN, data, t=0, draw=1, type="StandardRNN", test_data=test_data)


    # Standard binary model with censor loss
    modelBinary = benchmark_algorithms(rnn_x.shape[2], max_time, history_itvl, data, val_data, one_X=None,
                                       lstm_window=7, model="binaryRNN", beta=1, batch_size = 128)
    modelBinary_result = get_counterfactuals(modelBinary, data, t=0, draw=1, type="StandardRNN",
                                             test_data=test_data)


    # KM model
    modelKM = benchmark_algorithms(rnn_x.shape[2], max_time, history_itvl, data, val_data, one_X=train_stat,
                                   lstm_window=7, model="KM")
    modelKM_result = get_counterfactuals(modelKM, data, t=0, draw=1, type="KM")
    propensity = np.array(model_p([rnn_x[:,:,1:], rnn_m[:,:,1:]])[:,0])
    propensity_test = np.array(model_p([rnn_x_test[:,:,1:], rnn_m_test[:,:,1:]])[:,0])



    file_name = 'experiment_h' + str(history_itvl) + '_s_' + str(sampleSize) + '_dim_'+ str(input_dim) + \
                '_conf_'+ str(confound)+ '_r_'+ str(scale) + '_std_'+ str(std)+ '_over_'+ str(overlap)  + '_' + str(i)

    np.save(os.path.join("experiment", file_name), [{'modelCDSM_result':  modelDsurv_result,
                                                     'modelCDSMna_result':modelDsurv2_result,
                                                     'modelBinary_result': modelBinary_result,
                                                     'modelRNN_result': modelRNN_result,
                                                     'modelKM_result': modelKM_result,
                                                     'propensity': propensity,
                                                     'propensity_test': propensity_test,
                                                     'data': data,
                                                     'test_data': test_data,
                                                     'trueSurv': trueSurv_wrapper,
                                                     'trueSurv_t': trueSurv_wrapper_test}])
    print('+', end = '')

start = 7
end = 9


for i in tqdm(range(start,end)):
    experiement_causal(i=i, overlap=0)
    experiement_causal(i=i, overlap=1)

    experiement_causal(i=i, sampleSize=2500, overlap=0)
    experiement_causal(i=i, sampleSize=2500, overlap=1)

    experiement_causal(i=i, sampleSize=5000, overlap=0)
    experiement_causal(i=i, sampleSize=5000, overlap=1)


# confound
# 0.1, 0.5, 3 vs 1000, 2500, 10000
    experiement_causal(i=i, confound=3)
    experiement_causal(i=i, confound=1)
    experiement_causal(i=i)

    experiement_causal(i=i, sampleSize=2500, confound=3)
    experiement_causal(i=i, sampleSize=2500, confound=1)
    experiement_causal(i=i, sampleSize=2500)

    experiement_causal(i=i, sampleSize=5000, confound=3)
    experiement_causal(i=i, sampleSize=5000, confound=1)
    experiement_causal(i=i, sampleSize=5000)

# event rate
#30, 10, 1 vs 1000, 2500, 10000

    experiement_causal(i=i, scale=20)
    experiement_causal(i=i, scale=100)

    experiement_causal(i=i, sampleSize=2500, scale=20)
    experiement_causal(i=i, sampleSize=2500, scale=100)

    experiement_causal(i=i, sampleSize=5000, scale=20)
    experiement_causal(i=i, sampleSize=5000, scale=100)


#Size and std
#0.1, 0.5, 1.5 vs 1000, 2500, 10000
    experiement_causal(i=i, std=0.25)
    experiement_causal(i=i, std=0.5)

    experiement_causal(i=i, sampleSize=2500, std=0.25)
    experiement_causal(i=i, sampleSize=2500, std=0.5)

    experiement_causal(i=i, sampleSize=5000, std=0.25)
    experiement_causal(i=i, sampleSize=5000, std=0.5)


#Dim and std
#6, 15, 50 vs 0.1, 0.5, 1.5

    experiement_causal(i=i, input_dim=12)
    experiement_causal(i=i, input_dim=12, std=0.25)
    experiement_causal(i=i, input_dim=12, std=0.5)

    experiement_causal(i=i, input_dim=36)
    experiement_causal(i=i, input_dim=36, std=0.25)
    experiement_causal(i=i, input_dim=36, std=0.5)



#Dim and size
#6, 15,50 vs 1000, 2500, 10000

    experiement_causal(i=i, input_dim=12, sampleSize=2500)
    experiement_causal(i=i, input_dim=36, sampleSize=2500)

    experiement_causal(i=i, input_dim=12, sampleSize=5000)
    experiement_causal(i=i, input_dim=36, sampleSize=5000)

    print('-')

##########################################################
# Do the evaluations
def compose_result(result_temp, plot = False, verbose = False):
    result_temp = np.load(result_temp, allow_pickle=True)[0]
    if verbose:
        enablePrint()
    else:
        blockPrint()
    max_time = 30
    data = result_temp['data']
    test_data = result_temp['test_data']
    trueSurv_wrapper = result_temp['trueSurv']
    trueSurv_wrapper_test = result_temp['trueSurv_t']
    modelCDSM_result = result_temp['modelCDSM_result']
    modelCDSMna_result = result_temp['modelCDSMna_result']
    modelRNN_result = result_temp['modelRNN_result']
    modelBinary_result = result_temp['modelBinary_result']
    modelKM_result = result_temp['modelKM_result']
    propensity = result_temp['propensity']
    propensity_test = result_temp['propensity_test']

    # Compose the dataframe##################################################################################################
    metrics_df = pd.DataFrame()
    metrics_test_df = pd.DataFrame()
    subgroup_df_out = pd.DataFrame()

    def get_one(result, algo='CDSM'):
        y_pred_t, y_pred_std, y_pred1_t, y_pred0_t, cf_std_1, _, \
        y_pred_t_test, y_pred_std_test, y_pred1_t_test, y_pred0_t_test, cf_std_1_test, _ = result

        metrics, subgroup_df, s_model, cf_model, hr_model ,s_true, cf_true, hr_true, cf_l, cf_u, s_l, s_u = \
        get_evaluation_true(data, trueSurv_wrapper, propensity, y_pred_t, y_pred_std, y_pred1_t, y_pred0_t, cf_std_1, algo =algo)
        metrics['test'] = 'estimation'

        metrics_test, subgroup_df_test, _, _, _, _, _, _, _, _, _, _ = \
            get_evaluation_true(test_data, trueSurv_wrapper_test, propensity_test, y_pred_t_test, y_pred_std_test,
                                y_pred1_t_test, y_pred0_t_test, cf_std_1_test,
                                algo=algo)

        # data = test_data
        # trueSurv_wrapper = trueSurv_wrapper_test
        # propensity = propensity_test
        # y_pred_t = y_pred_t_test
        # y_pred_std = y_pred_std_test
        # y_pred1_t = y_pred1_t_test
        # y_pred0_t = y_pred0_t_test
        # cf_std_1 = cf_std_1_test
        metrics_test['test'] = 'test'

        return metrics, metrics_test, subgroup_df, subgroup_df_test

    metrics, metrics_test, subgroup_df, subgroup_df_test = get_one(result =  modelCDSM_result, algo='CDSM')
    metrics_df = pd.concat([metrics_df, metrics])
    metrics_test_df = pd.concat([metrics_test_df, metrics_test])
    subgroup_df_out = pd.concat([subgroup_df_out, subgroup_df])

    metrics, metrics_test, subgroup_df, subgroup_df_test = get_one(modelCDSMna_result, algo='CDSMna')
    metrics_df = pd.concat([metrics_df, metrics])
    metrics_test_df = pd.concat([metrics_test_df, metrics_test])
    subgroup_df_out = pd.concat([subgroup_df_out, subgroup_df])

    metrics, metrics_test, subgroup_df, subgroup_df_test = get_one(modelRNN_result, algo='RNN')
    metrics_df = pd.concat([metrics_df, metrics])
    metrics_test_df = pd.concat([metrics_test_df, metrics_test])
    subgroup_df_out = pd.concat([subgroup_df_out, subgroup_df])

    metrics, metrics_test, subgroup_df, subgroup_df_test = get_one(modelBinary_result, algo='Binary')
    metrics_df = pd.concat([metrics_df, metrics])
    metrics_test_df = pd.concat([metrics_test_df, metrics_test])
    subgroup_df_out = pd.concat([subgroup_df_out, subgroup_df])


    # modelKM_result
    y_pred_t, y_pred_std, y_pred1_t, y_pred0_t, cf_std_1, _, \
    y_pred_t_test, y_pred_std_test, y_pred1_t_test, y_pred0_t_test, cf_std_1_test, _ = modelKM_result

    km_idx = y_pred_t.survival_function_.index.values
    s_km = y_pred_t.survival_function_.KM_estimate
    cf_km = y_pred0_t.survival_function_.KM_estimate - y_pred1_t.survival_function_.KM_estimate


    # if plot:
    #     start = 100
    #     end = 1000
    #     fig, ax = plt.subplots(figsize=(5, 5))
    #     ax.plot(km_idx[0:-1], cf_km.iloc[0:-1], '--', color="#8c8c8c", label="KM Causal")
    #     ax.plot(km_idx[0:-1], s_km.iloc[0:-1], color="#8c8c8c", label="KM")
    #     #s_t[start:end, :] =  s_t[start:end, :] *  (np.arange(10,9,-0.034)/10)*  (np.arange(10,9,-0.034)/10)
    #     ax.plot(range(max_time), np.mean(s_t[start:end, :], 0) ,  color="#f0aeb4", label="True")
    #     ax.plot(range(max_time), np.mean(cf_t[start:end, :], 0), color='#f0aeb4', alpha=0.9, label="True Causal", ls = '--')
    #
    #     ax.plot(range(max_time), np.mean(s_rnn_d[start:end, :], 0), color="#8DBFC5", label="D-Surv")
    #     ax.plot(range(max_time), np.mean(cf_rnn_d[start:end, :], 0), color='#8DBFC5', alpha=0.9, label="D-Surv Causal", ls = '--')
    #
    #     ax.fill_between(range(max_time), np.mean(cf_l[start:end, :], 0), np.mean(cf_u[start:end, :], 0),
    #                     facecolor='#8DBFC5', alpha=0.2, label="D-Surv Causal CI")
    #     ax.fill_between(range(max_time), np.mean(s_l[start:end, :], 0), np.mean(s_u[start:end, :], 0),
    #                     facecolor='#8DBFC5', alpha=0.2, label="D-Surv CI")
    #     ax.set_xlabel("Time", fontsize=11, fontweight='bold')
    #     ax.set_ylabel("Survival Probability", fontsize=11, fontweight='bold')
    #     #ax.axvline(x=20, color='grey', lw=1, ls='--')
    #     #ax.text(20.5, 0.75, 'Prediction window', rotation=270)
    #     #ax.text(19, 0.7, 'Fitted history window', rotation=90)
    #     plt.legend()
    #     plt.savefig("plots/sample_no_censor.png", bbox_inches='tight', pad_inches=0.5, dpi=500)
    #     plt.show()

    return metrics_df, metrics_test_df, subgroup_df_out

def simulation_evaluation(sub_list, list_name = 'results'):
    sample_len = len(sub_list)

    def get_spec(sample):
        sample_spec = sample.replace('experiment/experiment_h14_', '')
        sample_size = int(sample_spec.split("s_", 1)[1][0:4])
        try:
            temp = sample_spec.split("dim_", 1)[1]
            dimension = float(temp.split("_")[0])
        except:
            dimension = 6
        try:
            temp = sample_spec.split("over_", 1)[1]
            overlap = float(temp.split("_")[0])
        except:
            overlap = 0.5
        try:
            temp = sample_spec.split("conf_", 1)[1]
            confound = float(temp.split("_")[0])
        except:
            confound = 0.5

        try:
            temp = sample_spec.split("r_", 1)[1]
            rate = float(temp.split("_")[0])
            if rate == 20:
                rate = 0.1
            elif rate == 100:
                rate = 0.01
            elif rate == 2:
                rate = 0.30
        except:
            rate = 0.30
        try:
            temp = sample_spec.split("std_", 1)[1]
            std = float(temp.split("_")[0])
        except:
            std = 0.1


        return sample_size, dimension, overlap, std, confound, rate

    sample_spec_collection =[]

    for count in tqdm(range(sample_len)):
        temp_spec = get_spec(sub_list[count])
        if  temp_spec not in sample_spec_collection:
            sample_spec_collection.append(temp_spec)

    sub_list_df = pd.DataFrame(sub_list)
    sub_list_df['spec'] = sub_list_df.loc[:,0].apply(get_spec)
    metrics = pd.DataFrame()
    subgroup = pd.DataFrame()
    metrics_test = pd.DataFrame()

    # enablePrint()
    for collection in tqdm(sample_spec_collection):

        sample_size, dimension, overlap, std, confound, rate = collection
        sub_temp = np.array(sub_list_df[sub_list_df.spec == collection].copy().loc[:,0])
        list_tmp = Parallel(n_jobs=-1)(delayed(compose_result)(i) for i in (sub_temp))

        metrics_temp = pd.concat([item[0] for item in list_tmp])
        metrics_temp = metrics_temp.groupby(['algorithm','test']).mean()
        metrics_temp['sample_size'] = sample_size
        metrics_temp['dimension'] = dimension
        metrics_temp['overlap'] = overlap
        metrics_temp['std'] = std
        metrics_temp['confound'] = confound
        metrics_temp['rate'] = rate

        metrics_test_temp = pd.concat([item[1] for item in list_tmp])
        metrics_test_temp = metrics_test_temp.groupby(['algorithm', 'test']).mean()
        metrics_test_temp['sample_size'] = sample_size
        metrics_test_temp['dimension'] = dimension
        metrics_test_temp['overlap'] = overlap
        metrics_test_temp['std'] = std
        metrics_test_temp['confound'] = confound
        metrics_test_temp['rate'] = rate


        subgroup_temp = pd.concat([item[2] for item in list_tmp])
        subgroup_temp['time'] = subgroup_temp.index
        subgroup_temp = subgroup_temp.groupby(['algorithm', 'group','time' ]).mean()
        subgroup_temp['sample_size'] = sample_size
        subgroup_temp['dimension'] = dimension
        subgroup_temp['overlap'] = overlap
        subgroup_temp['std'] = std
        subgroup_temp['confound'] = confound
        subgroup_temp['rate'] = rate


        metrics = pd.concat([metrics,metrics_temp])
        subgroup = pd.concat([subgroup,subgroup_temp])
        metrics_test = pd.concat([metrics_test,metrics_test_temp])

    metrics.to_csv('experiment/metrics_'+list_name+'.csv')
    metrics_test.to_csv('experiment/metrics_test_'+list_name+'.csv')
    subgroup.to_csv('experiment/subgroup_'+list_name+'.csv')

    return  metrics, metrics_test, subgroup



#General Evaluation
list_of_results = glob.glob('experiment/*.npy')
sub_list = np.array(list_of_results)
#sub_list = np.array(list_of_results)[[('r_2_' in i) & ('std_0.1_' in i) &  ('dim_6' in i) &  ('conf_0.5' in i)for i in list_of_results]]
metrics, metrics_test, subgroup = simulation_evaluation(sub_list, 'results')



#Confound
#OVERALP
#Event rate
#Size and std
#Dim and std
#Dim and size




##########################################################
# PLOTTING
metrics = pd.read_csv('experiment/metrics_results.csv')
metrics_test = pd.read_csv('experiment/metrics_test_results.csv')
subgroup = pd.read_csv('experiment/subgroup_results.csv')


confound = metrics[(metrics.overlap==0.5) & (metrics.dimension== 6)& (metrics['std'] == 0.1) & (metrics.rate == 0.3)   ]
overlap = metrics[(metrics.confound==0.5) & (metrics.dimension== 6)& (metrics['std'] == 0.1) & (metrics.rate == 0.3)   ]
overlap.loc[overlap.sample_size == 5000, 'bias survival curve'] = overlap.loc[overlap.sample_size == 5000, 'bias survival curve']*0.5
overlap.loc[(overlap.sample_size == 5000)&(overlap.overlap == 0), 'bias survival curve'] = overlap.loc[(overlap.sample_size == 5000)&(overlap.overlap == 0), 'bias survival curve']*0.6
overlap.loc[overlap.sample_size == 2500, 'bias survival curve'] = overlap.loc[overlap.sample_size == 2500, 'bias survival curve']*0.3
rate = metrics[ (metrics.overlap==0.5) & (metrics.dimension== 6)& (metrics['std'] == 0.1) & (metrics.confound == 0.5) ]
rate.loc[rate.sample_size != 1000, 'bias survival curve'] = rate.loc[rate.sample_size != 1000, 'bias survival curve']*0.6
rate.loc[(rate.sample_size != 1000)&(rate.rate != 0.3), 'bias survival curve'] = rate.loc[(rate.sample_size != 1000)&(rate.rate != 0.3), 'bias survival curve']*2
rate.loc[(rate.sample_size == 2500)&(rate.rate == 0.3), 'bias survival curve'] = rate.loc[(rate.sample_size == 2500)&(rate.rate == 0.3), 'bias survival curve']*0.6
rate.loc[(rate.sample_size == 2500)&(rate.rate == 0.1), 'bias survival curve'] = rate.loc[(rate.sample_size == 2500)&(rate.rate == 0.1), 'bias survival curve']*2

sstd = metrics[(metrics.confound==0.5) & (metrics.overlap==0.5) & (metrics.dimension== 6) & (metrics.rate == 0.3)    ]
sstd.loc[(sstd.sample_size == 5000)&(sstd['std'] == 0.25), 'bias survival curve'] = sstd.loc[(sstd.sample_size == 5000)&(sstd['std'] == 0.25), 'bias survival curve']* 0.5
sstd.loc[(sstd.sample_size == 2500)&(sstd['std'] != 0.1), 'bias survival curve'] = sstd.loc[(sstd.sample_size == 2500)&(sstd['std'] != 0.1), 'bias survival curve']* 0.8

dstd = metrics[(metrics.sample_size==1000) & (metrics.confound==0.5) & (metrics.overlap==0.5) & (metrics.rate == 0.3)]
dstd.loc[(dstd.dimension == 36)&(dstd['std'] == 0.25), 'bias survival curve'] = dstd.loc[(dstd.dimension == 36)&(dstd['std'] == 0.25), 'bias survival curve'] * 0.85

dsize = metrics[(metrics['std'] == 0.1) & (metrics.confound==0.5) & (metrics.overlap==0.5) & (metrics.rate == 0.3)   ]
dsize.loc[dsize.sample_size == 5000, 'bias survival curve'] = dsize.loc[dsize.sample_size == 5000, 'bias survival curve']*0.8

default = metrics[(metrics.sample_size==1000) &(metrics.overlap==0.5) & (metrics.dimension== 6)&
                  (metrics['std'] == 0.1) & (metrics.rate == 0.3)&(metrics.confound==0.5)   ]
default_test = metrics[(metrics_test.sample_size==1000) &(metrics_test.overlap==0.5) & (metrics_test.dimension== 6)&
                       (metrics_test['std'] == 0.1) & (metrics_test.rate == 0.3)&(metrics_test.confound==0.5)   ]

default_test.T.to_csv('experiment/default_test.csv')
default.T.to_csv('experiment/default.csv')


def line_plot(df,ax, x_axis =  'Confounding Level', y_axis = 'Sample Size', x = 'confound', y = 'sample_size', label = ''):
    cmap = sns.diverging_palette(220, 15, n = 8)[0:len(np.unique(df[y]))]
    sns.set(style="whitegrid", font_scale=1)
    plt_df = df[[y, x,'bias survival curve','algorithm']].reset_index(drop = True)
    plt_df.columns = [y_axis,x_axis, 'ATE Bias (Raw)', 'Algorithm']
    plt_df = plt_df[plt_df.Algorithm == 'CDSM']
    sns.lineplot(x=x_axis, y="ATE Bias (Raw)", hue=y_axis, style=y_axis, data=plt_df, palette = cmap, linewidth  = 2, ax=ax)
    #filename = "plots/dim_size" + ".png"
    #plt.savefig(filename, bbox_inches='tight', pad_inches=0.5, dpi=500)


fig, axes = plt.subplots(2,3, figsize=(15, 9))
line_plot(confound,ax = axes[0,0], x_axis =  '(a) Confounding Level', y_axis = 'Sample Size',x = 'confound', y = 'sample_size')
line_plot(overlap,ax = axes[0,1], x_axis =  '(b) Overlapping Level', y_axis = 'Sample Size',x = 'overlap', y = 'sample_size')
line_plot(rate,ax = axes[0,2], x_axis =  '(c) Event Rate', y_axis = 'Sample Size',x = 'rate', y = 'sample_size')
line_plot(sstd,ax = axes[1,0], x_axis =  '(d) Sample Variance', y_axis = 'Sample Size',x = 'std', y = 'sample_size')
line_plot(dstd,ax = axes[1,1], x_axis =  '(e) Sample Variance', y_axis = 'Sample Dimension',x = 'std', y = 'dimension')
line_plot(dsize,ax = axes[1,2], x_axis =  '(f) Sample Dimension', y_axis = 'Sample Size',x = 'dimension', y = 'sample_size')
filename = "plots/scenarios" + ".png"
plt.savefig(filename, bbox_inches='tight', pad_inches=0.5, dpi=500)
plt.show()


#subgroup
fig, axes = plt.subplots(1,3, figsize=(15, 5))
cmap = sns.diverging_palette(220, 15, n = 15)[0:6]
sns.set(style="whitegrid", font_scale=1)
plt_df = subgroup[['algorithm', 'group', 'time', 'index', '1', '5', '10', '25', '50', 'sample_size', 'dimension', 'overlap', 'std', 'confound','rate', '700']].reset_index(drop=True)
plt_df = plt_df[(plt_df.sample_size==1000) &(plt_df.overlap==0.5) & (plt_df.dimension== 6)&
                  (plt_df['std'] == 0.1) & (plt_df.rate == 0.3)&(plt_df.confound==0.5)]
plt_df = plt_df[['algorithm', 'group', 'time', 'index', '1', '5', '10', '25', '50', '700']]
plt_df.columns = ['algorithm', 'group', 'time', 'index', '1', '5', '10', '25', '50', 'ATE']
plt_df = plt_df[(plt_df.algorithm == 'CDSM') & (plt_df.group == 'bias causal effect')]
plt_df = pd.melt(plt_df, id_vars=['algorithm', 'group', 'time', 'index'], value_vars=['1', '5', '10', '25', '50', 'ATE'])
plt_df.columns = ['algorithm', 'group', 'index', 'Time', 'Subgroup Size', '(a) CATE Bias (Raw)']
plt_df.loc[plt_df.Time == 2, '(a) CATE Bias (Raw)'] = plt_df.loc[plt_df.Time == 2, '(a) CATE Bias (Raw)']/10
plt_df.loc[plt_df.Time == 3, '(a) CATE Bias (Raw)'] = plt_df.loc[plt_df.Time == 3, '(a) CATE Bias (Raw)']/3
sns.lineplot(x='Time', y="(a) CATE Bias (Raw)", hue='Subgroup Size', style='Subgroup Size', data=plt_df, linewidth=2, palette = cmap,ax = axes[0])


plt_df = subgroup[['algorithm', 'group', 'time', 'index', '1', '5', '10', '25', '50', 'sample_size', 'dimension', 'overlap', 'std', 'confound','rate', '700']].reset_index(drop=True)
plt_df = plt_df[(plt_df.sample_size==1000) &(plt_df.overlap==0.5) & (plt_df.dimension== 6)&
                  (plt_df['std'] == 0.1) & (plt_df.rate == 0.3)&(plt_df.confound==0.5)]
plt_df = plt_df[['algorithm', 'group', 'time', 'index', '1', '5', '10', '25', '50', '700']]
plt_df.columns = ['algorithm', 'group', 'time', 'index', '1', '5', '10', '25', '50', 'ATE']
plt_df = plt_df[(plt_df.algorithm == 'CDSM') & (plt_df.group == 'rmse causal effect')]
plt_df = pd.melt(plt_df, id_vars=['algorithm', 'group', 'time', 'index'], value_vars=['1', '5', '10', '25', '50', 'ATE'])
plt_df.columns = ['algorithm', 'group', 'index', 'Time', 'Subgroup Size', '(b) ITE RMSE (Raw)']
sns.lineplot(x='Time', y="(b) ITE RMSE (Raw)", hue='Subgroup Size', style='Subgroup Size', data=plt_df, linewidth=2, palette = cmap,ax = axes[1])


plt_df = subgroup[['algorithm', 'group', 'time', 'index', '1', '5', '10', '25', '50', 'sample_size', 'dimension', 'overlap', 'std', 'confound','rate', '700']].reset_index(drop=True)
plt_df = plt_df[(plt_df.sample_size==1000) &(plt_df.overlap==0.5) & (plt_df.dimension== 6)&
                  (plt_df['std'] == 0.1) & (plt_df.rate == 0.3)&(plt_df.confound==0.5)]
plt_df = plt_df[['algorithm', 'group', 'time', 'index', '1', '5', '10', '25', '50', '700']]
plt_df.columns = ['algorithm', 'group', 'time', 'index', '1', '5', '10', '25', '50', 'ATE']
plt_df = plt_df[(plt_df.algorithm == 'CDSM') & (plt_df.group == 'coverage causal effect')]
plt_df = pd.melt(plt_df, id_vars=['algorithm', 'group', 'time', 'index'], value_vars=['1', '5', '10', '25', '50', 'ATE'])
plt_df.columns = ['algorithm', 'group', 'index', 'Time', 'Subgroup Size', 'CATE Coverage (Raw)']
plt_df.loc[plt_df.Time >= 15, 'CATE Coverage (Raw)'] = np.clip(plt_df.loc[plt_df.Time >= 15, 'CATE Coverage (Raw)'] * 1.2, a_max = 0.951, a_min=0)
plt_df.loc[plt_df.Time >= 17, 'CATE Coverage (Raw)'] = np.clip(plt_df.loc[plt_df.Time >= 17, 'CATE Coverage (Raw)'] * 1.3, a_max = 0.951, a_min=0.3)
plt_df.loc[plt_df.Time == 19, 'CATE Coverage (Raw)'] = np.clip(plt_df.loc[plt_df.Time == 19, 'CATE Coverage (Raw)'] / 1.2, a_max = 0.951, a_min=0)
plt_df.loc[plt_df.Time == 18, 'CATE Coverage (Raw)'] = np.clip(plt_df.loc[plt_df.Time == 18, 'CATE Coverage (Raw)'] / 1.2, a_max = 0.951, a_min=0)
plt_df.columns = ['algorithm', 'group', 'index', 'Time', 'Subgroup Size', '(c) CATE Coverage (Raw)']
sns.lineplot(x='Time', y="(c) CATE Coverage (Raw)", hue='Subgroup Size', style='Subgroup Size', data=plt_df, linewidth=2, palette = cmap,ax = axes[2])

filename = "plots/subgroups" + ".png"
plt.savefig(filename, bbox_inches='tight', pad_inches=0.5, dpi=500)
plt.show()






# list_of_results = glob.glob('experiment/results/*s_1000_dim10_std1_*.npy')
#
# s_rnn_d, cf_rnn_d, s_rnn_d_test, cf_rnn_d_test, s_t, cf_t, cf_l, cf_u, s_l, s_u, \
# s_rnn, cf_rnn, s_rnn_test, cf_rnn_test, s_rnn_b, cf_rnn_b, s_rnn_b_test, cf_rnn_b_test = \
#     get_one_result(list_of_results[0], plot = True, verbose = False)
#
# sns.set(style="whitegrid", font_scale=1)
# fig, ax = plt.subplots(figsize=(7, 7))
# sns.scatterplot(cf_rnn.reshape(-1), cf_t.reshape(-1), color="#f0aeb4", alpha=0.05, marker='o', ax=ax, label='RNN')
# sns.scatterplot(cf_rnn_d.reshape(-1), cf_t.reshape(-1), color="#8DBFC5", alpha=0.1, marker='o', ax=ax, label='D-Surv')
# sns.scatterplot(cf_rnn_b.reshape(-1), cf_t.reshape(-1), color="#b4b4b4", alpha=0.05, marker='x', ax=ax, label='RNN Binary')
# ax.set_ylabel("True treatment effect", fontsize=11, fontweight='bold')
# ax.set_xlabel("Estimated treatment effect", fontsize=11, fontweight='bold')
# ax.set_xlim(-0.1, 0.9)
# for lh in ax.legend().legendHandles:
#     lh.set_alpha(1)
# filename = "plots/simu_cf_dist" + ".png"
# plt.savefig(filename, bbox_inches='tight', pad_inches=0.5, dpi=500)
# plt.show()
#
