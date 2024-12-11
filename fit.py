import numpy as np
import scipy
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import sys
import optuna
import time

def compute_mcfadden(NLL, sid):
    null_log_likelihood = 0
    human = pd.read_pickle(f"data/human.pkl").query("sid==@sid")
    trials = human['trial'].unique()
    stages = human['stage'].unique()
    for trial in trials:
        for stage in stages:
            act = human.query("trial==@trial & stage==@stage")['action'].unique()[0]
            prob = scipy.special.expit(0)
            null_log_likelihood -= np.log(prob) if act==1 else np.log(1-prob)
    mcfadden_r2 = 1 - NLL/null_log_likelihood
    # n_trials = len(subdata['trial'].unique()) * len(subdata['stage'].unique())
    # mcfadden_r2 = 1 - NLL/(n_trials*np.log(0.5))
    return mcfadden_r2

def get_param_names(model_type):
    if model_type in ['NEF-WM', 'NEF-RL']:
        param_names = ['type', 'inv_temp']
    if model_type == 'RL1':
        param_names = ['type', 'learning_rate_1', 'inv_temp']
    if model_type == 'RL3':
        param_names = ['type', 'learning_rate_1', 'learning_rate_2', 'learning_rate_3', 'inv_temp']
    if model_type == 'RL3rd':
        param_names = ['type', 'learning_rate_1', 'learning_rate_2', 'learning_rate_3', 'inv_temp']
    if model_type in ['Z0', 'Z05']:
        param_names = ['type', 'inv_temp']
    if model_type == 'Z':
        param_names = ['type', 'z', 'inv_temp']
    if model_type == 'ZK':
        param_names = ['type', 'z', 'k', 'inv_temp']
    if model_type == 'ZS':
        param_names = ['type', 's1', 's2', 's3', 'z', 'inv_temp']
    if model_type == 'DGn':
        param_names = ['type', 'inv_temp']
    if model_type == 'DGrd':
        param_names = ['type', 'inv_temp']
    if model_type == 'DGrds':
        param_names = ['type', 's2', 's3', 'inv_temp']
    if model_type == 'DGrdp':
        param_names = ['type', 'inv_temp']
    if model_type == 'DGrdpz':
        param_names = ['type', 'z', 'inv_temp']
    return param_names

def get_expectation(model_type, params, trial, stage, sid):
    human = pd.read_pickle(f"data/human.pkl").query("sid==@sid")
    if model_type in ['NEF-WM', 'NEF-RL']:
        if model_type == 'NEF-WM':
            nef_data = pd.read_pickle(f"data/WM_loadzk_estimates.pkl").query("sid==@sid")
        if model_type == 'NEF-RL':
            nef_data = pd.read_pickle(f"data/RL_loadzk_estimates.pkl").query("sid==@sid")
        expectation = nef_data.query("trial==@trial & stage==@stage")['estimate'].to_numpy()[-1]
    if model_type in ['RL1', "RL3", "RL3rd"]:
        if model_type == 'RL1':
            learning_rates = [1, params[0], params[0], params[0]]
        if model_type in ['RL3', 'RL3rd']:
            learning_rates = [1, params[0], params[1], params[2]]
        subdata = human.query("trial==@trial & stage<=@stage")
        observations = subdata['color'].to_numpy()
        RDs = subdata['RD'].to_numpy()
        expectation = 0
        for o, obs in enumerate(observations):
            stg = int(subdata.iloc[o]['stage'])
            RD = RDs[o] if (model_type in ['RL3rd'] and stg>1) else 1
            learning_rate = learning_rates[stg]
            error = obs - expectation
            LR = RD*learning_rate
            LR = np.clip(LR, 0, 1)
            expectation += LR * error
    if model_type in ['Z', 'Z0', 'Z05', 'ZK']:
        if model_type in ['Z', 'ZK']:
            z = params[0]
        elif model_type=='Z0':
            z = 0
        elif model_type=='Z05':
            z = 0.5
        if model_type=='ZK':
            k = params[1]
        else:
            k = 1
        subdata = human.query("trial==@trial & stage<=@stage")
        observations = subdata['color'].to_numpy()
        RDs = subdata['RD'].to_numpy()
        expectation = 0
        for o, obs in enumerate(observations):
            stg = int(subdata.iloc[o]['stage'])
            decay = 1 / (o+1)
            RD = 0 if stg in [0,1] else RDs[o]
            error = obs - expectation
            weight = decay**k + z*RD
            weight = np.clip(weight, 0, 1)
            expectation += weight * error
    if model_type=='ZS':
        decays = [1, params[0], params[1], params[2]]
        z = params[3]
        subdata = human.query("trial==@trial & stage<=@stage")
        observations = subdata['color'].to_numpy()
        RDs = subdata['RD'].to_numpy()
        expectation = 0
        neighbors = len(subdata['who'].unique()) - 1
        for o, obs in enumerate(observations):
            stg = int(subdata.iloc[o]['stage'])
            # decay = 1 if stg==0 else 1/(stg*neighbors)
            decay = decays[stg]
            RD = 0 if stg in [0,1] else RDs[o]
            error = obs - expectation
            weight = decay + z*RD
            weight = np.clip(weight, 0, 1)
            expectation += weight * error
        # print(stage, decay, observations, expectation)
    if model_type in ['DGn', 'DGrd', 'DGrds', 'DGrdp', 'DGrdpz']:
        subdata = human.query("trial==@trial & stage<=@stage")
        observations = subdata['color'].to_numpy()
        if model_type=='DGn':
            expectation = np.mean(observations)
        if model_type in ['DGrd', 'DGrds']:
            if model_type == 'DGrd':
                late_weights = [1,1]
            if model_type == 'DGrds':
                late_weights = [params[0], params[1]]                
            if stage in [0,1]:
                expectation = np.mean(observations)
            else:
                early_data = human.query("trial==@trial & stage<=1")
                early_observations = early_data['color'].to_numpy()       
                expectation = np.mean(early_observations)
                late_data = human.query("trial==@trial & stage>1")
                late_observations = late_data['color'].to_numpy()
                RDs = late_data['RD'].to_numpy()
                for o, obs in enumerate(late_observations):
                    stg = 0 if int(late_data.iloc[o]['stage'])==2 else 1
                    w = late_weights[stg]
                    RD = RDs[o]
                    # weight = np.clip(w*RD, 0, 1)
                    expectation += w*RD*obs
        if model_type in ['DGrdp', 'DGrdpz']:
            RDs = subdata['RD'].to_numpy()
            w = 1 if model_type=='DGrdp' else params[0]
            weights = []
            for o, obs in enumerate(observations):
                stg = int(subdata.iloc[o]['stage'])
                RD = RDs[o] if stg>1 else 0
                weights.append(1+w*RD)  # weight of observation is 1 by default, but 1+RD for stage 2-3
            expectation = np.mean(weights*observations) # average weighted observations
    return expectation


def likelihood(params, model_type, sid):
    NLL = 0
    human = pd.read_pickle(f"data/human.pkl").query("sid==@sid")
    trials = human['trial'].unique()
    stages = human['stage'].unique()
    inv_temp = params[-1]
    for trial in trials:
        for stage in stages:
            expectation = get_expectation(model_type, params, trial, stage, sid)
            act = human.query("trial==@trial and stage==@stage")['action'].unique()[0]
            prob = scipy.special.expit(inv_temp*expectation)
            # print(f'trial {trial}, stage {stage}, expectation {expectation}, action {act}, prob {prob}')
            NLL -= np.log(prob) if act==1 else np.log(1-prob)
    # print(params, model_type, sid, NLL)
    return NLL

def stat_fit_scipy(model_type, sid, save=True):
    if model_type in ['NEF-WM', 'NEF-RL']:
        param0 = [1.0]
        bounds = [(0,10)]
    if model_type == 'RL1':
        param0 = [0.1, 1.0]
        bounds = [(0,1), (0,10)]
    if model_type == 'RL3':
        param0 = [0.1, 0.1, 0.1, 1.0]
        bounds = [(0,1), (0,1), (0, 1), (0,10)]
    if model_type == 'RL3rd':
        param0 = [0.5, 0.5, 0.5, 1.0]
        bounds = [(0,10), (0,10), (0, 10), (0,10)]
    if model_type in ['Z0', 'Z05']:
        param0 = [1.0]
        bounds = [(0,10)]
    if model_type == 'Z':
        param0 = [0.5, 1.0]
        bounds = [(0,2), (0,10)]
    if model_type == 'ZK':
        param0 = [0.5, 1.0, 1.0]
        bounds = [(0,2), (0.1,2), (0,10)]
    if model_type == 'ZS':
        param0 = [0.3, 0.2, 0.1, 0.5, 1.0]
        bounds = [(0,1), (0,1), (0,1), (0,2), (0,10)]
    if model_type == 'DGn':
        param0 = [1.0]
        bounds = [(0,10)]
    if model_type == 'DGrd':
        param0 = [1.0]
        bounds = [(0,10)]
    if model_type == 'DGrds':
        param0 = [0.5, 0.5, 1]
        bounds = [(0, 3), (0,3), (0,10)]
    if model_type == 'DGrdp':
        param0 = [1.0]
        bounds = [(0,10)]
    if model_type == 'DGrdpz':
        param0 = [1.0, 1.0]
        bounds = [(0.5, 2.0), (0,10)]
    result = scipy.optimize.minimize(
        fun=likelihood,
        x0=param0,
        args=(model_type, sid),
        bounds=bounds,
        options={'disp':False})
    # Save Negative Log Likelihood and McFadden R2
    NLL = result.fun
    mcfadden_r2 = compute_mcfadden(NLL, sid)
    performance_data = pd.DataFrame([[model_type, sid, NLL, mcfadden_r2]], columns=['type', 'sid', 'NLL', 'McFadden R2'])
    if save:
        performance_data.to_pickle(f"data/{model_type}_{sid}_performance.pkl")
    # Save the fitted parameters
    param_names = get_param_names(model_type)
    params = list(result.x)
    params.insert(0, model_type)
    fitted_params = pd.DataFrame([params], columns=param_names)
    if save:
        fitted_params.to_pickle(f"data/{model_type}_{sid}_params.pkl")
    return performance_data, fitted_params

def stat_fit_optuna(model_type, sid, optuna_trials=100, save=True):
    study = optuna.create_study(study_name=f"{model_type}_{sid}", direction="minimize")
    study.optimize(lambda trial: optuna_wrapper(trial, model_type, sid), n_trials=optuna_trials)
    # Save Negative Log Likelihood and McFadden R2
    NLL = study.best_value
    mcfadden_r2 = compute_mcfadden(NLL, sid)
    performance_data = pd.DataFrame([[model_type, sid, NLL, mcfadden_r2]], columns=['type', 'sid', 'NLL', 'McFadden R2'])
    if save:
        performance_data.to_pickle(f"data/{model_type}_{sid}_performance.pkl")
    # Save the fitted parameters
    param_names = get_param_names(model_type)
    params = study.best_trial.params
    params = np.insert(params, 0, model_type)
    fitted_params = pd.DataFrame([params], columns=param_names)
    if save:
        fitted_params.to_pickle(f"data/{model_type}_{sid}_params.pkl")
    return performance_data, fitted_params


def optuna_wrapper(trial, model_type, sid):
    params = []
    if model_type in ['NEF-WM', 'NEF-RL']:
        params.append(trial.suggest_float("inv_temp", 0, 10, step=0.01))
    if model_type == 'RL1':
        params.append(trial.suggest_float("learning_rate_1", 0, 1, step=0.001))
        params.append(trial.suggest_float("inv_temp", 0, 10, step=0.01))
    if model_type == 'RL3':
        params.append(trial.suggest_float("learning_rate_1", 0, 1, step=0.001))
        params.append(trial.suggest_float("learning_rate_2", 0, 1, step=0.001))
        params.append(trial.suggest_float("learning_rate_3", 0, 1, step=0.001))
        params.append(trial.suggest_float("inv_temp", 0, 10, step=0.01))
    if model_type == 'RL3rd':
        params.append(trial.suggest_float("learning_rate_1", 0, 10, step=0.01))
        params.append(trial.suggest_float("learning_rate_2", 0, 10, step=0.01))
        params.append(trial.suggest_float("learning_rate_3", 0, 10, step=0.01))
        params.append(trial.suggest_float("inv_temp", 0, 10, step=0.01))
    if model_type == 'ZK':
        params.append(trial.suggest_float("z", 0, 2, step=0.01))
        params.append(trial.suggest_float("k", 0, 2, step=0.01))
        params.append(trial.suggest_float("inv_temp", 0, 10, step=0.01))
    if model_type == 'DGn':
        params.append(trial.suggest_float("inv_temp", 0, 10, step=0.01))
    if model_type == 'DGrd':
        params.append(trial.suggest_float("inv_temp", 0, 10, step=0.01))
    if model_type == 'DGrds':
        params.append(trial.suggest_float("s2", 0, 10, step=0.01))
        params.append(trial.suggest_float("s3", 0, 10, step=0.01))
        params.append(trial.suggest_float("inv_temp", 0, 10, step=0.01))
    if model_type == 'DGrdp':
        params.append(trial.suggest_float("inv_temp", 0, 10, step=0.01))
    if model_type == 'DGrdpz':
        params.append(trial.suggest_float("z", 0.5, 2.0, step=0.01))
        params.append(trial.suggest_float("inv_temp", 0, 10, step=0.01))
    NLL = likelihood(params, model_type, sid)
    return NLL

if __name__ == '__main__':

    model_type = sys.argv[1]
    sid = int(sys.argv[2])
    # method = sys.argv[3]
    method = 'scipy'
    # method = 'optuna'

    start = time.time()
    if method=='scipy':
        if model_type=='all':
            model_types = ['RL1', 'RL3rd', 'DGn', 'ZK', 'NEF-WM', 'NEF-RL']
            for mt in model_types:
                print(f"fitting {mt}, sid {sid}")
                performance_data, fitted_params = stat_fit_scipy(mt, sid)
        else:
            print(f"fitting {model_type}, {sid}")
            performance_data, fitted_params = stat_fit_scipy(model_type, sid)

    if method=='optuna':
        if model_type=='all':
            model_types = ['RL1', 'RL3rd', 'DGn', 'ZK', 'NEF-WM', 'NEF-RL']
            for mt in model_types:
                print(f"fitting {mt}, sid {sid}")
                performance_data, fitted_params = stat_fit_optuna(mt, sid)
        else:
            print(f"fitting {model_type}, {sid}")
            performance_data, fitted_params = stat_fit_optuna(model_type, sid)

    print(performance_data)
    print(fitted_params)
    end = time.time()
    print(f"runtime {(end-start)/60:.4} min")