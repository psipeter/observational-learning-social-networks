import numpy as np
import scipy
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import sys

def likelihood(param, model_type, sid):
    NLL = 0
    data = pd.read_pickle(f"data/behavior.pkl").query("sid==@sid")
    trials = data['trial'].unique()
    stages = data['stage'].unique()
    if model_type in ['RL1', 'RL1rd']:
        alpha = param[0]  # learning rate for all stages
        inv_temp = param[1]  # determines randomness in policy (passed to softmax)
    if model_type in ['RL2', 'RL2rd']:
        alpha = param[0]  # learning rate for stage 1
        beta = param[1]  # learning rate for stages 2-3
        inv_temp = param[2]  # determines randomness in policy (passed to softmax)
    if model_type in ['NEF-WM', 'NEF-RL']:
        inv_temp = param[0]  # determines randomness in policy (passed to softmax)
        datafile = "WM_z05k12" if model_type=='NEF-WM' else "RL_z05k12"
        nef_data = pd.read_pickle(f"data/{datafile}.pkl").query("type!='human' & sid==@sid")
    for trial in trials:
        for stage in stages:
            subdata = data.query("trial==@trial & stage==@stage")
            if model_type in ['NEF-WM', "NEF-RL"]:
                nef_subdata = nef_data.query("trial==@trial & stage==@stage")
                expectation = nef_subdata['estimate'].to_numpy()[-1]
            else:
                observations = subdata['color'].to_numpy()
                RDs = subdata['RD'].to_numpy()
                if stage==0:
                    expectation = observations[0]
                else:
                    learning_rate = beta if (model_type in ['RL2', 'RL2rd'] and stage>1) else alpha
                    for n in range(len(observations)):
                        obs = observations[n]
                        RD = RDs[n] if (model_type in ['RL1rd', 'RL2rd'] and stage>1) else 1
                        error = obs - expectation
                        LR = np.clip(RD*learning_rate, 0, 1)
                        expectation += LR * error               
            act = subdata['action'].unique()[0]
            prob = scipy.special.expit(inv_temp*expectation)
            # print(f'stage {stage}, expectation {expectation}, action {act}, alpha {alpha}, inv-temp {inv_temp}, prob {prob}')
            NLL -= np.log(prob) if act==1 else np.log(1-prob)
    return NLL

def stat_fit(model_type, sid, save=True):
    dfs = []
    columns = ['type', 'sid', 'neg-log-likelihood', 'alpha', 'beta', 'inv-temp']
    if model_type in ['NEF-WM', 'NEF-RL']:
        param0 = [10.0]
        bounds = [(0,100)]
    if model_type == 'RL1':
        param0 = [0.1, 10]
        bounds = [(0,1), (0,100)]
    if model_type == 'RL1rd':
        param0 = [0.5, 10]
        bounds = [(0,10), (0,100)]
    if model_type == 'RL2':
        param0 = [0.1, 0.1, 10]
        bounds = [(0,1), (0,1), (0,100)]
    if model_type == 'RL2rd':
        param0 = [0.5, 0.5, 10]
        bounds = [(0,10), (0,10), (0,100)]
    # if model_type in ['RL1', 'RL1rd']:
    #     param0 = [0.1, 10]
    #     bounds = [(0,1), (0,100)]
    # if model_type in ['RL2', 'RL2rd']:
    #     param0 = [0.1, 0.1, 10.0]
    #     bounds = [(0,1), (0,1), (0,100)]
    result = scipy.optimize.minimize(
        fun=likelihood,
        x0=param0,
        args=(model_type, sid),
        bounds=bounds,
        options={'disp':False})
    NLL = result.fun
    params = result.x
    if model_type in ['NEF-RL', 'NEF-WM']:
        alpha = None
        beta = None
        inv_temp = params[0]
    if model_type in ['RL1', 'RL1rd']:
        alpha = params[0]
        beta = None
        inv_temp = params[1]
    if model_type in ['RL2', 'RL2rd']:
        alpha = params[0]
        beta = params[1]
        inv_temp = params[2]
    fitted = pd.DataFrame([[model_type, sid, NLL, alpha, beta, inv_temp]], columns=columns)
    if save:
        fitted.to_pickle(f"data/{model_type}_{sid}.pkl")
    return fitted

if __name__ == '__main__':

    model_type = sys.argv[1]
    sid = int(sys.argv[2])
    fitted = stat_fit(model_type, sid)