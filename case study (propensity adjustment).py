import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
# from lifelines import CoxTimeVaryingFitter
# from lifelines.utils import add_covariate_to_timeline
# from lifelines.utils import to_long_format
from lifelines import KaplanMeierFitter

from utilss.data_simulator import get_data
from utilss.model import create_model, get_counterfactuals

print("simulating...")

max_time = 30
data, pred_data, val_data, surv_func_wrapper, train_full, val_full, test_full, one_X = \
    get_data(input_dim=10, sampleSize=1000, max_time=max_time, prediction_itvl=1, history_itvl=14,
             overlap=1, seed=1234, confound=0.1, scale=5)

print("simulation completed")


s_true = np.cumprod(surv_func_wrapper[0],1)
cf_true = np.cumprod(surv_func_wrapper[2],1) - np.cumprod(surv_func_wrapper[1],1)
hr_true = surv_func_wrapper[2] / surv_func_wrapper[1]


# Train the model
rnn_x, rnn_m, rnn_s, rnn_y, Time = data #estimation data set

# Model fitting lstm model with censor loss
# Set gamma at 0 for no selection bias loss function

kmf0 = KaplanMeierFitter().fit(one_X[one_X.A == 0]['T'], event_observed=one_X[one_X.A == 0]['Y'])
kmf1 = KaplanMeierFitter().fit(one_X[one_X.A == 1]['T'], event_observed=one_X[one_X.A == 1]['Y'])
kmf = KaplanMeierFitter().fit(one_X['T'], event_observed=one_X['Y'])
s_km = kmf.survival_function_.KM_estimate
cf_km = kmf1.survival_function_.KM_estimate - kmf0.survival_function_.KM_estimate


modelCDSM, model_p, history_dict = create_model(rnn_x.shape[2], max_time = max_time, history_itvl = 14,
                                                data = data, val_data=val_data, lstm_window= 7,
                                                alpha= 1, beta= 0.5, gamma= 0, load=False, verbose=0,
                                                model_name='CDSM(unadjusted)',
                                                batch_size= int(rnn_x.shape[0]/2.2), layers = 3)

modelCDSM = get_counterfactuals(modelCDSM, data, t=0, draw=1, test_data=pred_data)

y_pred_t, y_pred_std, y_pred1_t, y_pred0_t, cf_std_1, _, \
y_pred_t_test, y_pred_std_test, y_pred1_t_test, y_pred0_t_test, cf_std_1_test, _ = modelCDSM
s_durv = np.cumprod(y_pred_t, 1)
cf_durv = np.cumprod(y_pred0_t, 1) - np.cumprod(y_pred1_t, 1)
hr_durv = np.clip(y_pred1_t, 0.001, 1) / np.clip(y_pred0_t, 0.001, 1)


#############################################
#############################################
sns.set(style="whitegrid", font_scale=1)
fig, ax = plt.subplots(figsize=(7, 7))
ax.plot(range(max_time), np.mean(s_true[:, 0:max_time], 0), color="#8c8c8c", label="True", alpha=0.9)
ax.plot(range(max_time - 1), s_km, '--', color="#888888", label="KM")
ax.plot(range(max_time), np.mean(s_durv, 0), color='#8DBFC5', label="CDSM(unadjusted)")
ax.fill_between(range(max_time), np.quantile(s_durv,0.05,0), np.quantile(s_durv,0.95,0), color='#8DBFC5', alpha=0.2)
ax.fill_between(range(max_time), np.quantile(s_true,0.05,0), np.quantile(s_true,0.95,0), color='#8c8c8c', alpha=0.2)
#ax.set_xticklabels(np.arange(-8, max_time * 3, 8))
ax.set_xlabel("Time", fontsize=11, fontweight='bold')
ax.set_ylabel("Survival Probability", fontsize=11, fontweight='bold')
plt.legend()
plt.show()




sns.set(style="whitegrid", font_scale=1)
fig, ax = plt.subplots(figsize=(7, 7))
ax.plot(range(max_time), np.mean(cf_true[:, 0:max_time], 0), color="#8c8c8c", label="True")
ax.plot(range(max_time), np.mean(cf_durv, 0),  color='#8DBFC5', alpha=0.9, label="CDSM(unadjusted)")
ax.fill_between(range(max_time), np.quantile(cf_durv,0.05,0), np.quantile(cf_durv,0.95,0), color='#8DBFC5', alpha=0.2)
#ax.set_xticklabels(np.arange(-8, max_time * 3, 8))
ax.set_xlabel("Time", fontsize=11, fontweight='bold')
ax.set_ylabel("ATE (Difference in Survival Probability)", fontsize=11, fontweight='bold')
plt.legend()
plt.show()















