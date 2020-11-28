





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

