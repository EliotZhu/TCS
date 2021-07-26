import gc,os,sys,psutil

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
tf.random.set_seed(1234)

from utilss.model import create_model, get_counterfactuals, fit_model
from utilss.data_simulator import get_data
from lifelines import KaplanMeierFitter


def blockPrint():
    sys.stdout = open(os.devnull, 'w')


def enablePrint():
    sys.stdout = sys.__stdout__

path = os.getcwd()
##########################################################
# Run the experiments
def print_km(df):
    kmf = KaplanMeierFitter(label="AF")
    kmf.fit(df[df['A'] == 1]['T'], df[df['A'] == 1]['Y'])
    kmf1 = KaplanMeierFitter(label="AF2")
    kmf1.fit(df[df['A'] == 0]['T'], df[df['A'] == 0]['Y'])
    plt.plot(np.array(kmf.survival_function_)[0:25], color='green')
    plt.plot(np.array(kmf1.survival_function_)[0:25], color='grey')
    plt.show()
    return kmf, kmf1


def experiement_causal(i, simu_dim=6, history_itvl=5, sampleSize=1500, std=0.1, confound=0.1, scale=2, dynamic=False):

    max_time = 10
    file_name = 'experiment_h' + str(history_itvl) + '_s_' + str(sampleSize) + '_dim_' + str(simu_dim) + \
                '_conf_' + str(confound) + '_r_' + str(scale) + '_std_' + str(std) + '_dynamic_' + str(
                dynamic) + '_' + str(i)

    if not os.path.exists(os.path.join(path, "Documents/scripts/Python/CDSM/experiment", file_name + ".npy")):

        # max_time = 10
        # simu_dim = 6
        # history_itvl = 5
        # sampleSize = 3000
        # confound = 0.1
        # scale = 2
        # std = 0.25
        #

        proc = psutil.Process(os.getpid())
        gc.collect()
        mem0 = proc.memory_info().rss

        surv_func_wrapper, surv_func_wrapper_test, train, val, test = \
            get_data(input_dim=simu_dim,
                     sampleSize=sampleSize,
                     max_time=max_time,
                     history_itvl=history_itvl,
                     seed=np.random.random_integers(1, 1000),
                     std=std,
                     confound=confound, scale=scale, dynamic=dynamic)

        # kmf, kmf1 = print_km(train)

        def get_true(df, A=None):
            y1_true_grid = np.zeros((len(np.unique(df.ID)), max_time))
            h1_grid = np.zeros((len(np.unique(df.ID)), max_time))
            p_series = []
            for i in np.unique(df['T1']):
                pt = np.array(np.sum(df[df['T1'] == i][['' + str(i) for i in np.arange(1, simu_dim + 1)]], 1))
                p_series.append(pt.reshape(len(pt), 1))
            p_series = np.concatenate(p_series, 1)
            p_series = np.mean(p_series, 1)

            if isinstance(A, int):
                Ak = A
            else:
                Ak = df.groupby(['ID']).apply(lambda x: np.mean(x['A']))

            for k in range(0, max_time):
                t_effect = (k) ** (1 / scale)
                y1_true_grid[:, k] = (p_series * t_effect + Ak) / 10
                h1_grid[:, k] = k * 0.001 + y1_true_grid[:, k]

            true_potential_outcomes = np.array([np.exp(- h) for h in h1_grid])
            true_potential_outcomes[:, 0] = 1
            true_potential_outcomes = np.cumprod(true_potential_outcomes, 1)
            return true_potential_outcomes

        trueSurv_wrapper_1 = get_true(test, 1)
        trueSurv_wrapper_0 = get_true(test, 0)

        # Dynamic survival model
        model = create_model(simu_dim, history_itvl, max_time)
        best_models = fit_model(model, max_time, simu_dim=simu_dim, history_itvl=history_itvl,
                                train=train.astype('float32'), val=val.astype('float32'))
        result = get_counterfactuals(best_models, test, simu_dim=simu_dim, max_time=max_time,
                                                history_itvl=history_itvl)

        surv_ind_0, surv_ind_1, surv_ind_std_0, surv_ind_std_1, ite, ite_std, ate, ate_std, hr, hr_std = result

        mem1 = proc.memory_info().rss
        pd = lambda x2, x1: 100.0 * (x2 - x1) / mem0
        print("Allocation: %0.2f%%" % pd(mem1, mem0))

        # plt.plot(ate)
        # plt.plot(np.mean(true_potential_outcomes_1 - true_potential_outcomes_0, 0), color='green')
        # # plt.plot(np.array(kmf.survival_function_)[0:25] - np.array(kmf1.survival_function_)[0:25], color='grey')
        # plt.show()
        #
        # plt.plot(np.mean(surv_ind_0, 0), color='grey')
        # plt.plot(np.mean(true_potential_outcomes_0, 0), '--', color='grey')
        # plt.plot(np.mean(surv_ind_1, 0), color='blue')
        # plt.plot(np.mean(true_potential_outcomes_1, 0), '--', color='blue')
        # plt.show()
        np.save(os.path.join(path,"Documents/scripts/Python/CDSM/experiment", file_name), [{'modelCDSM_result': result,
                                                         'trueSurv_1': trueSurv_wrapper_1,
                                                         'trueSurv_0': trueSurv_wrapper_0}])
        print('+', end='')


start = 0
end = 8

for i in range(start, end):
    # confound and size
    # 0.1, 0.5, 3 vs 1000, 2500, 10000
    experiement_causal(i=i, confound=0.3)
    experiement_causal(i=i, confound=0.2)
    experiement_causal(i=i, confound=0)
    experiement_causal(i=i)

    experiement_causal(i=i, confound=0.3, sampleSize=3000)
    experiement_causal(i=i, confound=0.2, sampleSize=3000)
    experiement_causal(i=i, confound=0, sampleSize=3000)
    experiement_causal(i=i, sampleSize=3000)

    experiement_causal(i=i, confound=0.3, sampleSize=10000)
    experiement_causal(i=i, confound=0.2, sampleSize=10000)
    experiement_causal(i=i, confound=0, sampleSize=10000)
    experiement_causal(i=i, sampleSize=10000)

    experiement_causal(i=i, dynamic=True)
    experiement_causal(i=i, sampleSize=10000, dynamic=True)
    experiement_causal(i=i, sampleSize=3000, dynamic=True)


    # Size and std
    # 0.1, 0.5, 1.5 vs 1000, 2500, 10000
    experiement_causal(i=i, std=0.25)
    experiement_causal(i=i, std=0.5)
    experiement_causal(i=i, std=1)


    experiement_causal(i=i, sampleSize=3000, std=0.25)
    experiement_causal(i=i, sampleSize=3000, std=0.5)
    experiement_causal(i=i, sampleSize=3000, std=1)


    experiement_causal(i=i, sampleSize=10000, std=0.25)
    experiement_causal(i=i, sampleSize=10000, std=0.5)
    experiement_causal(i=i, sampleSize=10000, std=1)


    # Dim and size
    # 6, 15,50 vs 1000, 2500, 10000
    experiement_causal(i=i, simu_dim=10, sampleSize=1500)
    experiement_causal(i=i, simu_dim=20, sampleSize=1500)
    experiement_causal(i=i, simu_dim=30, sampleSize=1500)

    experiement_causal(i=i, simu_dim=10, sampleSize=3000)
    experiement_causal(i=i, simu_dim=20, sampleSize=3000)
    experiement_causal(i=i, simu_dim=30, sampleSize=3000)

    experiement_causal(i=i, simu_dim=10, sampleSize=10000)
    experiement_causal(i=i, simu_dim=20, sampleSize=10000)
    experiement_causal(i=i, simu_dim=30, sampleSize=10000)

    print('-')

