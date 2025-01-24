import numpy as np
import scipy
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import sys
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
    return mcfadden_r2

def get_param_names(model_type):
    if model_type in ['RLz', 'NEF_RL']:
        param_names = ['type', 'sid', 'z', 'b', 'inv_temp']
    if model_type in ['DGz', 'NEF_WM']:
        param_names = ['type', 'sid', 'z', 'inv_temp']
    if model_type in ['DGn']:
        param_names = ['type', 'sid', 'inv_temp']
    if model_type in ['RL']:
        param_names = ['type', 'sid', 'alpha']
    return param_names

def get_param_init_bounds(model_type):
    if model_type in ['RLz', 'NEF_RL']:
        param0 = [0.1, 0.1, 1.0]
        bounds = [(0,3), (0,1), (0,30)]
    if model_type in ['DGz', 'NEF_WM']:
        param0 = [0.5, 1.0]
        bounds = [(0,3), (0,30)]
    if model_type in ['DGn']:
        param0 = [1.0]
        bounds = [(0,30)]
    if model_type in ['RL']:  # Carrabin
        param0 = [0.5]
        bounds = [(0,1)]        
    return param0, bounds

def get_expectations_carrabin(model_type, params, sid, trial, stage):
    human = pd.read_pickle(f"data/carrabin.pkl").query("sid==@sid")
    if model_type == 'bayes':
        subdata = human.query("trial==@trial & stage<=@stage")
        reds = subdata.query("color==1")['color'].size
        p_star = (reds+1)/(stage+2)
        expectation = 2*p_star-1
    if model_type == 'bayesPE':
        subdata = human.query("trial==@trial & stage<=@stage")
        p_star = 0.5
        for stg in range(stage):
            s_old = stg
            s_new = stg + 1
            color = subdata.query("stage==@s_new")['color'].unique()[0]
            delta = (1-p_star)/(s_old+3) if color==1 else -p_star/(s_old+3)
            p_star += delta
        expectation = 2*p_star-1
    if model_type == 'RL':
        subdata = human.query("trial==@trial & stage<=@stage")
        colors = subdata['color'].to_numpy()
        expectation = 0
        alpha = params[0]
        for color in colors:
            error = color - expectation
            expectation += alpha*error
    return expectation

def get_expectations(model_type, params, trial, stage, sid, noise=False, sigma=0, rng=np.random.RandomState(seed=0)):
    human = pd.read_pickle(f"data/human.pkl").query("sid==@sid")
    if model_type == 'NEF_WM':
        nef_data = pd.read_pickle(f"data/NEF_WM_{sid}_estimates.pkl")
        expectations = nef_data.query("trial==@trial & stage<=@stage")['estimate'].to_numpy()
    if model_type == 'NEF_RL':
        nef_data = pd.read_pickle(f"data/NEF_RL_{sid}_estimates.pkl")
        expectations = nef_data.query("trial==@trial & stage<=@stage")['estimate'].to_numpy()
    if model_type == 'RLz':
        z = params[0]
        b = params[1]
        learning_rates = [1, b, b, b]
        subdata = human.query("trial==@trial & stage<=@stage")
        observations = subdata['color'].to_numpy()
        RDs = subdata['RD'].to_numpy()
        expectation = 0
        expectations = []
        for o, obs in enumerate(observations):
            stg = int(subdata.iloc[o]['stage'])
            RD = 0 if stg<2 else RDs[o]
            learning_rate = learning_rates[stg]
            error = obs - expectation
            LR = learning_rate + z*RD
            LR = np.clip(LR, 0, 1)
            expectation += LR * error
            expectations.append(expectation)
    if model_type == 'DGz':
        z = params[0]
        subdata = human.query("trial==@trial & stage<=@stage")
        observations = subdata['color'].to_numpy()
        RDs = subdata['RD'].to_numpy()
        expectation = 0
        expectations = []
        for o, obs in enumerate(observations):
            stg = int(subdata.iloc[o]['stage'])
            decay = 1 / (o+1)
            RD = 0 if stg in [0,1] else RDs[o]
            error = obs - expectation
            weight = decay + z*RD
            weight = np.clip(weight, 0, 1)
            expectation += weight * error
            expectations.append(expectation)
    if model_type == 'DGn':
        subdata = human.query("trial==@trial & stage<=@stage")
        observations = subdata['color'].to_numpy()
        expectation = 0
        expectations = []
        for o, obs in enumerate(observations):
            decay = 1 / (o+1)
            error = obs - expectation
            weight = decay
            if noise:
                weight = rng.uniform((1-sigma)*weight, (1+sigma)*weight)
            weight = np.clip(weight, 0, 1)
            expectation += weight * error
            expectations.append(expectation)
    return expectations

def RMSE(params, model_type, sid):  # Carrabin loss function
    human = pd.read_pickle(f"data/carrabin.pkl").query("sid==@sid")
    trials = human['trial'].unique()
    stages = human['stage'].unique()
    errors = []
    for trial in trials:
        for stage in stages:
            expectation = get_expectations_carrabin(model_type, params, sid, trial, stage)
            response = human.query("trial==@trial and stage==@stage")['response'].unique()[0]
            errors.append(np.square(response - expectation))
    rmse = np.sqrt(np.mean(errors))
    return rmse

def likelihood(params, model_type, sid, noise=False, sigma=0):
    NLL = 0
    human = pd.read_pickle(f"data/human.pkl").query("sid==@sid")
    trials = human['trial'].unique()
    stages = human['stage'].unique()
    inv_temp = params[-1]
    for trial in trials:
        for stage in stages:
            expectations = get_expectations(model_type, params, trial, stage, sid, noise=noise, sigma=sigma)
            final_expectation = expectations[-1]
            act = human.query("trial==@trial and stage==@stage")['action'].unique()[0]
            prob = scipy.special.expit(inv_temp*final_expectation)
            NLL -= np.log(prob) if act==1 else np.log(1-prob)
    return NLL

def fit_carrabin(model_type, sid):
    param0, bounds = get_param_init_bounds(model_type)
    result = scipy.optimize.minimize(
        fun=RMSE,
        x0=param0,
        args=(model_type, sid),
        bounds=bounds,
        options={'disp':False})
    rmse = result.fun
    # Save Results and Best Fit Parameters
    performance_data = pd.DataFrame([[model_type, sid, rmse]], columns=['type', 'sid', 'RMSE'])
    performance_data.to_pickle(f"data/{model_type}_{sid}_performance.pkl")
    param_names = get_param_names(model_type)
    params = list(result.x)
    params.insert(0, sid)
    params.insert(0, model_type)
    fitted_params = pd.DataFrame([params], columns=param_names)
    fitted_params.to_pickle(f"data/{model_type}_{sid}_params.pkl")
    return performance_data, fitted_params

def stat_fit_scipy(model_type, sid):
    param0, bounds = get_param_init_bounds(model_type)
    result = scipy.optimize.minimize(
        fun=likelihood,
        x0=param0,
        args=(model_type, sid),
        bounds=bounds,
        options={'disp':False})
    NLL = result.fun
    mcfadden_r2 = compute_mcfadden(NLL, sid)
    # Save Results and Best Fit Parameters
    performance_data = pd.DataFrame([[model_type, sid, NLL, mcfadden_r2]], columns=['type', 'sid', 'NLL', 'McFadden R2'])
    performance_data.to_pickle(f"data/{model_type}_{sid}_performance.pkl")
    param_names = get_param_names(model_type)
    params = list(result.x)
    params.insert(0, sid)
    params.insert(0, model_type)
    fitted_params = pd.DataFrame([params], columns=param_names)
    fitted_params.to_pickle(f"data/{model_type}_{sid}_params.pkl")
    return performance_data, fitted_params

if __name__ == '__main__':
    model_type = sys.argv[1]
    sid = int(sys.argv[2])
    start = time.time()
    print(f"fitting {model_type}, {sid}")
    if model_type in ['RL']:
        performance_data, fitted_params = fit_carrabin(model_type, sid)
    else:
        performance_data, fitted_params = stat_fit_scipy(model_type, sid)
    print(performance_data)
    print(fitted_params)
    end = time.time()
    print(f"runtime {(end-start)/60:.4} min")