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
    if model_type=='RL1':
        alpha = param[0]  # learning rate for all stages
        inv_temp = param[1]  # determines randomness in policy (passed to softmax)
    if model_type=='RL2':
        alpha = param[0]  # learning rate for stage 1
        beta = param[1]  # learning rate for stages 2-3
        inv_temp = param[2]  # determines randomness in policy (passed to softmax)
    if model_type=='NEF-WM':
        inv_temp = param[0]  # determines randomness in policy (passed to softmax)
        nef_data = pd.read_pickle(f"data/WM_z05k12.pkl").query("type=='model-WM' & sid==@sid")
    if model_type=="NEF-RL":
        inv_temp = param[0]  # determines randomness in policy (passed to softmax)
        nef_data = pd.read_pickle(f"data/RL_z05k12.pkl").query("type=='model-RL' & sid==@sid")
    for trial in trials:
        for stage in stages:
            if model_type in ['NEF-WM', "NEF-RL"]:
                expectation = nef_data.query("trial==@trial & stage==@stage")['estimate'].to_numpy()[-1]
            else:
                observations = data.query("trial==@trial & stage==@stage")['color'].to_numpy()
                if stage==0:
                    expectation = observations[0]
                elif stage==1:
                    learning_rate = alpha
                    for obs in observations:
                        error = obs - expectation
                        expectation += learning_rate * error
                else:
                    learning_rate = alpha if model_type=='RL1' else beta
                    for obs in observations:
                        error = obs - expectation
                        expectation += learning_rate * error                
            act = data.query("trial==@trial & stage==@stage")['action'].unique()[0]
            prob = scipy.special.expit(inv_temp*expectation)
            print('stage', stage, 'expectation', expectation, 'action', act, 'prob', prob)
            NLL -= np.log(prob) if act==1 else np.log(1-prob)
        raise
    return NLL

def rerun(fitted, model_type, sid):
    data = pd.read_pickle(f"data/behavior.pkl").query("sid==@sid")
    params = fitted.query("type==@model_type & sid==@sid")
    trials = data['trial'].unique()
    stages = data['stage'].unique()
    if model_type=='RL1':
        alpha = params['alpha'].unique()[0]  # learning rate for all stages
        inv_temp = params['inv-temp'].unique()[0]  # determines randomness in policy (passed to softmax)
    if model_type=='RL2':
        alpha = params['alpha'].unique()[0]  # learning rate for stage 1
        beta = params['beta'].unique()[0]  # learning rate for stages 2-3
        inv_temp = params['inv-temp'].unique()[0]  # determines randomness in policy (passed to softmax)
    if model_type=='NEF-WM':
        inv_temp = params['inv-temp'].unique()[0]  # determines randomness in policy (passed to softmax)
        nef_data = pd.read_pickle(f"data/WM_z05k12.pkl").query("type=='model-WM' & sid==@sid")
    if model_type=="NEF-RL":
        inv_temp = params['inv-temp'].unique()[0]  # determines randomness in policy (passed to softmax)
        nef_data = pd.read_pickle(f"data/RL_z05k12.pkl").query("type=='model-RL' & sid==@sid")
    dfs = []
    columns = ['type', 'sid', 'trial', 'stage', 'color', 'estimate', 'prob']
    for trial in trials:
        for stage in stages:
            if model_type in ['NEF-WM', "NEF-RL"]:
                observations = data.query("trial==@trial & stage==@stage")['color'].to_numpy()
                expectations = nef_data.query("trial==@trial & stage==@stage")['estimate'].to_numpy()
                expectation = expectations[-1]
            else:
                observations = data.query("trial==@trial & stage==@stage")['color'].to_numpy()
                expectations = []
                if stage==0:
                    expectation = observations[0]
                    expectations.append(expectation.copy())
                elif stage==1:
                    learning_rate = alpha
                    for obs in observations:
                        error = obs - expectation
                        expectation += learning_rate * error
                        expectations.append(expectation.copy())
                else:
                    learning_rate = alpha if model_type=='RL1' else beta
                    for obs in observations:
                        error = obs - expectation
                        expectation += learning_rate * error
                        expectations.append(expectation.copy())
            prob = scipy.special.expit(inv_temp*expectation)  # only record prob at end of each stage (not for each obs)
            # print(observations, expectations)
            for i in range(len(observations)):
                E = expectations[i]
                O = observations[i]
                df = pd.DataFrame([[model_type, sid, trial, stage, O, E, prob]], columns=columns)
                dfs.append(df)
    rerun_data = pd.concat(dfs, ignore_index=True)
    return rerun_data

if __name__ == '__main__':

    model_type = sys.argv[1]
    sid = int(sys.argv[2])

    dfs = []
    columns = ['type', 'sid', 'neg-log-likelihood', 'alpha', 'beta', 'inv-temp']
    if model_type=='NEF-WM':
        param0 = [10]
        bounds = [(0,50)]
    if model_type=='NEF-RL':
        param0 = [10]
        bounds = [(0,50)]
    if model_type=='RL1':
        param0 = [0.1, 10]
        bounds = [(0,1), (0,50)]
    elif model_type=='RL2':
        param0 = [0.1, 0.1, 10]
        bounds = [(0,1), (0,1), (0,50)]
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
    if model_type=='RL1':
        alpha = params[0]
        beta = None
        inv_temp = params[1]
    elif model_type=='RL2':
        alpha = params[0]
        beta = params[1]
        inv_temp = params[2]
    fitted = pd.DataFrame([[model_type, sid, NLL, alpha, beta, inv_temp]], columns=columns)
    fitted.to_pickle(f"data/{model_type}_{sid}.pkl")