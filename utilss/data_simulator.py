# data_simulator.py
#
# Author: Jie Zhu
# Tested with Python version 3.8 and TensorFlow 2.5

from utilss.data_handler import LDataSimu


##########################################################
# Data Simulation
def get_data(input_dim=6,
             sampleSize=600,
             max_time=10,
             history_itvl=5,
             seed=123,
             std=0.1,
             confound=0.5,
             scale=2,
             dynamic=False):
    print('generating sample...', end='')
    train, surv_func_wrapper = LDataSimu(seed, sampleSize=sampleSize, max_time=max_time, history_itvl=history_itvl,
                                         simu_dim=input_dim, scale=scale,
                                         std=std, confound=confound, dynamic=False)
    print('....', end='')

    val, _ = LDataSimu(seed, sampleSize=1500, max_time=max_time, history_itvl=history_itvl,
                       simu_dim=input_dim, scale=scale,
                       std=std, confound=confound, dynamic=False)
    print('....', end='')

    test, surv_func_wrapper_test = LDataSimu(seed, sampleSize=2000, max_time=max_time, history_itvl=history_itvl,
                                             simu_dim=input_dim, scale=scale,
                                             std=std, confound=confound, dynamic=False)
    print('done!')

    # scale = 2  35%
    # scale = 10  10%
    # scale = 20  5%
    # scale = 100  1%
    # scale = 500  0.05%
    # index = train.groupby(['ID']).apply(lambda x: max(x.index))
    # one_X = train.iloc[np.array(index)].reset_index(drop=True).sort_values(['ID', 'T'])

    return surv_func_wrapper, \
           surv_func_wrapper_test, \
           train.sort_values(['ID', 'T']), \
           val.sort_values(['ID', 'T']), \
           test.sort_values(['ID', 'T'])
