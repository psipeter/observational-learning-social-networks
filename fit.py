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
    if model_type in ['ZK']:
        z = param[0]
        k = param[1]
        inv_temp = param[2]
    if model_type in ['NEF-WM', 'NEF-RL']:
        inv_temp = param[0]  # determines randomness in policy (passed to softmax)
        datafile = "WM_z05k12" if model_type=='NEF-WM' else "RL_z05k12"
        nef_data = pd.read_pickle(f"data/{datafile}.pkl").query("type!='human' & sid==@sid")
    for trial in trials:
        n_samples = 0
        expectation = 0
        for stage in stages:
            subdata = data.query("trial==@trial & stage==@stage")
            observations = subdata['color'].to_numpy()
            observations = 2*observations - np.ones_like(observations)
            RDs = subdata['RD'].to_numpy()
            if model_type in ['NEF-WM', "NEF-RL"]:
                nef_subdata = nef_data.query("trial==@trial & stage==@stage")
                expectation = nef_subdata['estimate'].to_numpy()[-1]
            elif model_type in ['RL1', "RL1rd", "RL2", "RL2rd"]:
                for n in range(len(observations)):
                    obs = observations[n]
                    if stage==0:
                        RD = 1
                        learning_rate = 1
                    if stage==1:
                        RD = 1
                        learning_rate = alpha
                    if stage in [2,3]:
                        RD = RDs[n] if model_type in ['RL1rd', 'RL2rd'] else 1
                        learning_rate = beta if model_type in ['RL2', 'RL2rd'] else alpha
                    error = obs - expectation
                    LR = RD*learning_rate
                    LR = np.clip(LR, 0, 1)
                    expectation += LR * error               
            elif model_type in ['ZK']:
                for n in range(len(observations)):
                    n_samples += 1
                    decay = 1 / n_samples
                    obs = observations[n]
                    RD = 0 if stage in [0,1] else RDs[n]
                    error = obs - expectation
                    weight = decay**k + z*RD
                    weight = np.clip(weight, 0, 1)
                    # print(decay, RD, z, k, weight)
                    expectation += weight * error                 
            act = subdata['action'].unique()[0]
            prob = scipy.special.expit(inv_temp*expectation)
            # print(f'stage {stage}, expectation {expectation}, action {act}, prob {prob}')
            NLL -= np.log(prob) if act==1 else np.log(1-prob)
    return NLL

def stat_fit(model_type, sid, save=True):
    dfs = []
    columns = ['type', 'sid', 'NLL']
    if model_type in ['NEF-WM', 'NEF-RL']:
        param0 = [1.0]
        bounds = [(0,100)]
    if model_type == 'RL1':
        param0 = [0.1, 1.0]
        bounds = [(0,1), (0,100)]
    if model_type == 'RL1rd':
        param0 = [0.5, 1.0]
        bounds = [(0,10), (0,100)]
    if model_type == 'RL2':
        param0 = [0.1, 0.1, 1.0]
        bounds = [(0,1), (0,1), (0,100)]
    if model_type == 'RL2rd':
        param0 = [0.5, 0.5, 1.0]
        bounds = [(0,10), (0,10), (0,100)]
    if model_type == 'ZK':
        param0 = [0.5, 1.0, 1.0]
        bounds = [(0,2), (0.1,2), (0,100)]
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
    fitted_params = result.x
    fitted_data = pd.DataFrame([[model_type, sid, NLL]], columns=columns)
    if save:
        fitted_data.to_pickle(f"data/{model_type}_{sid}.pkl")
        np.savez(f"data/{model_type}_{sid}.npz", fitted_params)
    return fitted_data, fitted_params

if __name__ == '__main__':

    model_type = sys.argv[1]
    sid = int(sys.argv[2])
    fitted_data, fitted_params = stat_fit(model_type, sid)