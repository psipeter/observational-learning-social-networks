import numpy as np
import scipy
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import sys
import time
import optuna
from NEF_RL import run_RL
from NEF_WM2 import run_WM
from scipy.stats import gaussian_kde

def NEF_carrabin_loss(trial, model_type, sid):
    if model_type=='NEF_RL':
        alpha = trial.suggest_float("alpha", 0.01, 1.0, step=0.01)
        n_error = trial.suggest_int("n_error", 10, 400, step=10)
        data = run_RL("carrabin", sid, alpha, z=0, n_error=n_error)
        # n_learning = trial.suggest_int("n_learning", 20, 400, step=20)
        # data = run_RL("carrabin", sid, alpha, z=0, n_learning=n_learning, n_error=n_error)
    if model_type=='NEF_WM':
        alpha = trial.suggest_float("alpha", 0.01, 1.0, step=0.01)
        n_memory = trial.suggest_int("n_memory", 10, 400, step=10)
        data = run_WM("carrabin", sid, alpha, z=0, n_memory=n_memory)
        # n_error = trial.suggest_int("n_error", 20, 400, step=20)
        # data = run_WM("carrabin", sid, alpha, z=0, n_memory=n_memory, n_error=n_error)
    loss = qid_abs_loss([], model_type, sid)
    return loss

def NEF_jiang_loss(trial, model_type, sid):
    if model_type=='NEF_RL':
        alpha = trial.suggest_float("alpha", 0.01, 1.0, step=0.01)
        # n_learning = trial.suggest_int("n_learning", 20, 400, step=20)
        n_error = trial.suggest_int("n_error", 20, 400, step=20)
        z = trial.suggest_float("z", 0.01, 1.0, step=0.01)
        inv_temp = trial.suggest_float("inv_temp", 0.01, 10, step=0.01)
        data = run_RL("jiang", sid, alpha, z=z, n_error=n_error)
    if model_type=='NEF_WM':
        alpha = trial.suggest_float("alpha", 0.01, 1.0, step=0.01)
        n_memory = trial.suggest_int("n_memory", 20, 400, step=20)
        # n_error = trial.suggest_int("n_error", 20, 400, step=20)
        z = trial.suggest_float("z", 0.01, 1.0, step=0.01)
        inv_temp = trial.suggest_float("inv_temp", 0.01, 10, step=0.01)
        data = run_WM("jiang", sid, alpha, z=z, n_memory=n_memory)
    NLL = likelihood([inv_temp], model_type, sid)
    return NLL

def compute_mcfadden(NLL, sid):
    null_log_likelihood = 0
    human = pd.read_pickle(f"data/jiang.pkl").query("sid==@sid")
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
    if model_type in ['RLz']:
        param_names = ['type', 'sid', 'z', 'b', 'inv_temp']
    if model_type in ['DGz']:
        param_names = ['type', 'sid', 'z', 'inv_temp']
    if model_type in ['DGn']:
        param_names = ['type', 'sid', 'inv_temp']
    if model_type in ['RL']:
        param_names = ['type', 'sid', 'mu']
    if model_type in ['RL_n', 'RL_n2']:
        param_names = ['type', 'sid', 'mu', 'sigma']
    if model_type in ['RL_nl', 'RL_ne']:
        param_names = ['type', 'sid', 'mu', 'sigma', 'k']
    if model_type in ['RL_nn']:
        param_names = ['type', 'sid', 'mu', 'sigma', 'v1']
    if model_type in ['bayes', 'bayesPE']:
        param_names = ['type', 'sid']
    if model_type in ['bayes_n']:
        param_names = ['type', 'sid', 'noise_n', 'noise_e']
    if model_type in ['NC', 'NC_ns']:
        param_names = ['type', 'sid', 'mu']
    if model_type in ['NC_n', 'NC_n2']:
        param_names = ['type', 'sid', 'mu', 'sigma']
    if model_type in ['NC_nn']:
        param_names = ['type', 'sid', 'mu', 'sigma', 'v1']
    if model_type in ['NC_nnn']:
        param_names = ['type', 'sid', 'mu', 'sigma', 'v1', 'v2']
    if model_type in ['NC_nln']:
        param_names = ['type', 'sid', 'mu', 'sigma', 'v1', 'v2']
    if model_type in ['NC_nll']:
        param_names = ['type', 'sid', 'mu', 'sigma', 'v1', 'v2']
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
    if model_type in ['bayes', 'bayesPE']:
        param0 = []
        bounds = []
    if model_type in ['bayes_n']:
        param0 = [0.1, 0.1]
        bounds = [(0,1), (0,1)]  
    if model_type in ['RL']:
        param0 = [0.5]
        bounds = [(0,1)]
    if model_type in ['RL_n']:
        param0 = [0.1, 0.05]
        bounds = [(0,1), (0.01, 0.1)]
    if model_type in ['RL_nl', 'RL_ne']:
        param0 = [0.1, 0.05, 0.0]
        bounds = [(0,1), (0.01, 0.1), (-1, 1)]
    if model_type in ['RL_n2']:
        param0 = [0.1, 0.05]
        bounds = [(0,1), (0.01,0.2)]
    if model_type in ['RL_nn']:
        param0 = [0.1, 0.0, 0.0]
        bounds = [(0,1), (0,0.01), (0,0.01)]
    if model_type in ['NC', 'NC_ns']:
        param0 = [0.1]
        bounds = [(0,1)]
    if model_type in ['NC_n', 'NC_n2']:
        param0 = [0.1, 0.05]
        bounds = [(0,1), (0.0, 0.1)]
    if model_type in ['NC_nn']:
        param0 = [0.1, 0.0, 0.0]
        bounds = [(0,1), (0,0.01), (0,0.01)]
    if model_type in ['NC_nnn']:
        param0 = [0.1, 0.0, 0.0, 0.0]
        bounds = [(0,1), (0,0.01), (0,0.01), (0,0.01)]
    if model_type in ['NC_nln']:
        param0 = [0.1, 0.0, 0.0, 0.0]
        bounds = [(0,1), (0,0.01), (0,0.01), (0,0.01)]
    if model_type in ['NC_nll']:
        param0 = [0.1, 0.0, 0.0, 0.0]
        bounds = [(0,1), (0,0.01), (0,0.01), (0,0.01)]
    return param0, bounds

def get_expectations_carrabin(model_type, params, sid, trial, stage, rng=np.random.RandomState(seed=0)):
    human = pd.read_pickle(f"data/carrabin.pkl").query("sid==@sid")
    if model_type == 'NEF_WM':
        nef_data = pd.read_pickle(f"data/NEF_WM_carrabin_{sid}_estimates.pkl")
        expectation = nef_data.query("trial==@trial & stage==@stage")['estimate'].unique()[0]
    if model_type == 'NEF_RL':
        nef_data = pd.read_pickle(f"data/NEF_RL_carrabin_{sid}_estimates.pkl")
        expectation = nef_data.query("trial==@trial & stage==@stage")['estimate'].unique()[0]
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
    if model_type == 'bayes_n':
        subdata = human.query("trial==@trial & stage<=@stage")
        expectation = 0
        noise_n = params[0]
        noise_e = params[1]
        for stg in range(stage):
            s_old = stg
            s_new = stg + 1
            weight = 1/(s_old+3)
            dW = rng.uniform((1-noise_n)*weight, (1+noise_n)*weight)
            weight = dW
            weight = np.clip(weight, 0, 1)
            color = subdata.query("stage==@s_new")['color'].unique()[0]
            error = color - expectation
            expectation += weight * error
            dE = rng.uniform((1-noise_e)*expectation, (1+noise_e)*expectation)
            expectation = dE
            expectation = np.clip(expectation, -1, 1)
    if model_type in ['RL', 'RL_n', 'RL_nl', 'RL_ne', 'RL_n2', 'RL_nn']:
        subdata = human.query("trial==@trial & stage<=@stage")
        colors = subdata['color'].to_numpy()
        expectation = 0
        for s, color in enumerate(colors):
            if model_type=='RL':
                LR = params[0]
                error = color - expectation
                expectation += LR*error
            elif model_type=='RL_n':
                LR = rng.normal(params[0], params[1])
                LR = np.clip(LR, 0, 1)
                error = color - expectation
                expectation += LR*error
            elif model_type=='RL_n2':
                LR = params[0]
                eps = rng.normal(0, params[1])
                error = color - expectation
                expectation += LR*error + eps
                expectation = np.clip(expectation, -1, 1)
            elif model_type=='RL_nl':
                LR = params[0] *(1 + s*params[2])  # LR changes linearly with stage
                LR = LR.clip(0, 1)
                eps = rng.normal(0, params[1])
                error = color - expectation
                expectation += LR*error + eps
                expectation = np.clip(expectation, -1, 1)
            elif model_type=='RL_ne':
                LR = params[0] * np.exp(s*params[2])  # LR changes by exponentially with stage
                LR = LR.clip(0, 1)
                eps = rng.normal(0, params[1])
                error = color - expectation
                expectation += LR*error + eps
                expectation = np.clip(expectation, -1, 1)
            elif model_type=='RL_nn':
                LR = rng.normal(params[0], params[1])
                rE = rng.normal(0, params[2])
                error = color - expectation
                expectation += LR*error + rE
                expectation = np.clip(expectation, -1, 1)
    if model_type in ['NC', 'NC_ns', 'NC_n', 'NC_n2', 'NC_nn', 'NC_nnn', 'NC_nln', 'NC_nll']:
        subdata = human.query("trial==@trial & stage<=@stage")
        colors = subdata['color'].to_numpy()
        p = 0.5
        r = 0.5
        for color in colors:
            if model_type == 'NC': # fixed learning rate, no response noise
                LR = params[0]
                r += LR*color
                p = r
                p = np.clip(p, 0, 1)
            if model_type == 'NC_ns': # fixed learning rate, no response noise
                LR = params[0]
                sigma = 0.04
                eps = rng.normal(0, np.square(sigma))
                r += LR*color + eps
                r = np.clip(r, -1, 1)
                p = r
            if model_type == 'NC_n': # normally distributed learning rate, no response noise
                LR = rng.normal(params[0], params[1])
                LR = np.clip(LR, 0, 1)
                r += LR*color
                p = r
                p = np.clip(p, 0, 1)
            if model_type == 'NC_n2': # fixed learning rate, normally distributed cognitive state noise, no response noise
                LR = params[0]
                eps = rng.normal(0, np.square(params[1]))
                r += LR*color + eps
                r = np.clip(r, 0, 1)
                p = r
            if model_type == 'NC_nn': # normally distributed learning rate, normally distributed response noise: Line 8
                LR = rng.normal(params[0], params[1])
                pE = rng.normal(0, params[2])
                r += LR*color
                p = r + pE
                p = np.clip(p, 0, 1)
            if model_type == 'NC_nnn': # normally distributed learning rate, normally distributed state AND response noise: Line 10
                LR = rng.normal(params[0], params[1])
                rE = rng.normal(0, params[2])
                pE = rng.normal(0, params[3])
                r += LR*color + rE
                p = r + pE
                p = np.clip(p, 0, 1)
            if model_type == 'NC_nln': # normally * log-normally distributed learning rate, normally distributed response noise: Line 11
                nLR = rng.normal(np.log(params[0]), params[1])
                lLR = np.exp(rng.normal(0, params[2]))
                pE = rng.normal(0, params[3])
                r += nLR*lLR*color
                p = r + pE
                p = np.clip(p, 0, 1)
            if model_type == 'NC_nll': # normally distributed learning rate, log-normally distributed state AND response noise: Line 12
                LR = rng.normal(params[0], params[1])
                rE = rng.normal(0, params[2])
                pE = np.exp(rng.normal(0, params[3]))
                r += LR*color + rE
                p += (r-p)*pE
                p = np.clip(p, 0, 1)
        expectation = 2*p-1
    return expectation

def get_expectations_jiang(model_type, params, trial, stage, sid, noise=False, sigma=0, rng=np.random.RandomState(seed=0)):
    human = pd.read_pickle(f"data/jiang.pkl").query("sid==@sid")
    if model_type == 'NEF_WM':
        nef_data = pd.read_pickle(f"data/NEF_WM_jiang_{sid}_estimates.pkl")
        expectations = nef_data.query("trial==@trial & stage<=@stage")['estimate'].to_numpy()
    if model_type == 'NEF_RL':
        nef_data = pd.read_pickle(f"data/NEF_RL_jiang_{sid}_estimates.pkl")
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

def kde_loss(params, model_type, sid):  # Carrabin loss function based on distribution of responses to each input sequence
    eval_points = np.linspace(-1, 1, 1000)
    human = pd.read_pickle(f"data/carrabin.pkl").query("sid==@sid")
    trials = human['trial'].unique()
    stages = human['stage'].unique()
    dfs = []
    columns = ['type', 'qid', 'response']
    for trial in trials:
        for stage in stages:
            qid = human.query("trial==@trial and stage==@stage")['qid'].unique()[0]
            response_human = human.query("trial==@trial and stage==@stage")['response'].unique()[0]
            dfs.append(pd.DataFrame([["human", qid, response_human]], columns=columns))
            response_model = get_expectations_carrabin(model_type, params, sid, trial, stage, rng=np.random.RandomState(seed=s+100*sid+1000*trial))
            dfs.append(pd.DataFrame([[model_type, qid, response_model]], columns=columns))
    response_data = pd.concat(dfs, ignore_index=True)
    total_loss = 0
    for qid in response_data['qid'].unique():
        n_total = response_data.query("type=='human'")['qid'].size
        n_qid = response_data.query("type=='human' & qid==@qid")['qid'].size
        responses_model = response_data.query("qid==@qid & type==@model_type")['response'].to_numpy()
        responses_human = response_data.query("qid==@qid & type=='human'")['response'].to_numpy()
        # convert these samples into probability distributions (smoothed histograms)
        # kernel density estimation is a way to estimate the probability density function (PDF) of a random variable in a non-parametric way
        # if model or human data has zero variance, calculate loss based on the difference between the means
        unique_model = response_data.query("qid==@qid & type==@model_type")['response'].unique()
        unique_human = response_data.query("qid==@qid & type=='human'")['response'].unique()
        if len(unique_model)==1 or len(unique_human)==1:
            kde_loss = np.abs(np.mean(responses_model) - np.mean(responses_human))
        else:
            kde_model = gaussian_kde(responses_model, bw_method='scott')
            kde_human = gaussian_kde(responses_human, bw_method='scott')
            samples_model = kde_model.evaluate(eval_points)
            samples_human = kde_human.evaluate(eval_points)
            samples_model = samples_model / np.sum(samples_model)
            samples_human = samples_human / np.sum(samples_human)
            kde_loss = np.mean(np.abs(samples_model - samples_human))  # ABS
            # kde_loss = np.sqrt(np.mean(np.square(samples_model - samples_human)))  # RMSE
            # add this to the total loss, weighed by the fraction of total trials
            W = n_qid / n_total
            total_loss += W * kde_loss
    print(params, total_loss)
    return total_loss

def qid_abs_loss(params, model_type, sid):  # Carrabin loss function based on distribution of responses to each input sequence
    human = pd.read_pickle(f"data/carrabin.pkl").query("sid==@sid")
    trials = human['trial'].unique()
    stages = human['stage'].unique()
    dfs = []
    columns = ['type', 'qid', 'response']
    for trial in trials:
        for stage in stages:
            qid = human.query("trial==@trial and stage==@stage")['qid'].unique()[0]
            response_human = human.query("trial==@trial and stage==@stage")['response'].unique()[0]
            dfs.append(pd.DataFrame([["human", qid, response_human]], columns=columns))
            response_model = get_expectations_carrabin(model_type, params, sid, trial, stage, rng=np.random.RandomState(seed=100*sid+1000*trial))
            dfs.append(pd.DataFrame([[model_type, qid, response_model]], columns=columns))
    response_data = pd.concat(dfs, ignore_index=True)
    total_loss = 0
    for qid in response_data['qid'].unique():
        n_total = response_data.query("type=='human'")['qid'].size
        n_qid = response_data.query("type=='human' & qid==@qid")['qid'].size
        # W = 1
        W = n_qid / n_total
        responses_model = response_data.query("qid==@qid & type==@model_type")['response'].to_numpy()
        responses_human = response_data.query("qid==@qid & type=='human'")['response'].to_numpy()
        total_loss += W * np.abs(np.mean(responses_model) - np.mean(responses_human))
        total_loss += W * np.abs(np.std(responses_model) - np.std(responses_human))
    print(params, total_loss)
    return total_loss

def RMSE(params, model_type, sid):  # Carrabin loss function
    human = pd.read_pickle(f"data/carrabin.pkl").query("sid==@sid")
    trials = human['trial'].unique()
    stages = human['stage'].unique()
    errors = []
    for trial in trials:
        for stage in stages:
            expectation = get_expectations_carrabin(model_type, params, sid, trial, stage, rng=np.random.RandomState(seed=sid+1000*trial))
            response = human.query("trial==@trial and stage==@stage")['response'].unique()[0]
            errors.append(np.square(response - expectation))
    rmse = np.sqrt(np.mean(errors))
    return rmse

def likelihood(params, model_type, sid, noise=False, sigma=0):
    NLL = 0
    human = pd.read_pickle(f"data/jiang.pkl").query("sid==@sid")
    trials = human['trial'].unique()
    stages = human['stage'].unique()
    inv_temp = params[-1]
    for trial in trials:
        for stage in stages:
            expectations = get_expectations_jiang(model_type, params, trial, stage, sid, noise=noise, sigma=sigma)
            final_expectation = expectations[-1]
            act = human.query("trial==@trial and stage==@stage")['action'].unique()[0]
            prob = scipy.special.expit(inv_temp*final_expectation)
            NLL -= np.log(prob) if act==1 else np.log(1-prob)
    return NLL

def fit_carrabin(model_type, sid, optuna_trials=2):
    if model_type in ['bayes', 'bayesPE']:
        params = []
        # rmse = RMSE(params, model_type, sid)
        loss = qid_abs_loss(params, model_type, sid)
        param_names = get_param_names(model_type)
        params.insert(0, sid)
        params.insert(0, model_type)
    elif model_type in ['NEF_RL', 'NEF_WM']:
        study = optuna.create_study(direction="minimize")
        study.optimize(lambda trial: NEF_carrabin_loss(trial, model_type, sid), n_trials=optuna_trials)
        best_params = study.best_trial.params
        loss = study.best_trial.value
        param_names = ["type", "sid"]
        params = [model_type, sid]
        for key, value in best_params.items():
            param_names.append(key)
            params.append(value)
        print(f"{len(study.trials)} trials completed. Best value is {loss:.4} with parameters:")
    else:
        param0, bounds = get_param_init_bounds(model_type)
        result = scipy.optimize.minimize(
            # fun=RMSE,
            # fun=kde_loss,
            fun=qid_abs_loss,
            x0=param0,
            args=(model_type, sid),
            bounds=bounds,
            method='L-BFGS-B',
            options={'maxiter': 10})
        loss = result.fun
        params = list(result.x)
        param_names = get_param_names(model_type)
        params.insert(0, sid)
        params.insert(0, model_type)

    # Save Results and Best Fit Parameters
    performance_data = pd.DataFrame([[model_type, sid, loss]], columns=['type', 'sid', 'loss'])
    performance_data.to_pickle(f"data/{model_type}_{dataset}_{sid}_performance.pkl")
    fitted_params = pd.DataFrame([params], columns=param_names)
    fitted_params.to_pickle(f"data/{model_type}_{dataset}_{sid}_params.pkl")
    return performance_data, fitted_params

def fit_jiang(model_type, sid, optuna_trials=400):
    if model_type in ['NEF_RL', 'NEF_WM']:
        study = optuna.create_study(direction="minimize")
        study.optimize(lambda trial: NEF_jiang_loss(trial, model_type, sid), n_trials=optuna_trials)
        best_params = study.best_trial.params
        NLL = study.best_trial.value
        param_names = ["type", "sid"]
        params = [model_type, sid]
        for key, value in best_params.items():
            param_names.append(key)
            params.append(value)
        print(f"{len(study.trials)} trials completed. Best value is {NLL:.4} with parameters:")
    else:
        param0, bounds = get_param_init_bounds(model_type)
        result = scipy.optimize.minimize(
            fun=likelihood,
            x0=param0,
            args=(model_type, sid),
            bounds=bounds,
            options={'disp':False})
        NLL = result.fun
        param_names = get_param_names(model_type)
        params = list(result.x)
        params.insert(0, sid)
        params.insert(0, model_type)
    # Save Results and Best Fit Parameters
    mcfadden_r2 = compute_mcfadden(NLL, sid)
    performance_data = pd.DataFrame([[model_type, sid, NLL, mcfadden_r2]], columns=['type', 'sid', 'NLL', 'McFadden R2'])
    performance_data.to_pickle(f"data/{model_type}_{dataset}_{sid}_performance.pkl")
    fitted_params = pd.DataFrame([params], columns=param_names)
    fitted_params.to_pickle(f"data/{model_type}_{dataset}_{sid}_params.pkl")
    return performance_data, fitted_params

if __name__ == '__main__':
    dataset = sys.argv[1]
    model_type = sys.argv[2]
    sid = int(sys.argv[3])
    start = time.time()
    print(f"fitting {model_type}, {sid}")
    if dataset=='carrabin':
        performance_data, fitted_params = fit_carrabin(model_type, sid)
    elif dataset=='jiang':
        performance_data, fitted_params = fit_jiang(model_type, sid)
    print(performance_data)
    print(fitted_params)
    end = time.time()
    print(f"runtime {(end-start)/60:.4} min")