import numpy as np
import scipy
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import sys
import time
import optuna
from NEF_RL import run_RL
from NEF_WM import run_WM
from NEF_syn import run_NEF_syn
from NEF_rec import run_NEF_rec
from scipy.stats import gaussian_kde

def NEF_carrabin_loss(trial, model_type, sid):
    if model_type=='NEF_RL':
        alpha = trial.suggest_float("alpha", 0.01, 1.0, step=0.01)
        n_all = trial.suggest_int("n_all", 10, 500, step=10)
        lambd = trial.suggest_float("lambda", 1.0, 1.0, step=0.01)
        data = run_RL("carrabin", sid, alpha=alpha, z=0, lambd=lambd, n_neurons=n_all, n_learning=n_all, n_error=n_all)
    if model_type=='NEF_WM':
        alpha = trial.suggest_float("alpha", 0.01, 1.0, step=0.01)
        n_all = trial.suggest_int("n_all", 10, 500, step=10)
        lambd = trial.suggest_float("lambda", 1.0, 1.0, step=0.01)
        data = run_WM("carrabin", sid, alpha=alpha, z=0, lambd=lambd, n_memory=n_all, n_neurons=n_all, n_error=n_all)
    if model_type=='NEF_syn':
        alpha = trial.suggest_float("alpha", 1e-5, 1e-3, step=1e-5)
        n_neurons = trial.suggest_int("n_neurons", 10, 1000, step=10)
        lambd = trial.suggest_float("lambda", 0.0, 0.0, step=0.01)
        data = run_NEF_syn("carrabin", sid, alpha=alpha, z=0, lambd=lambd, n_neurons=n_neurons)
    if model_type=='NEF_rec':
        alpha = trial.suggest_float("alpha", 0.01, 1.0, step=0.01)
        n_neurons = trial.suggest_int("n_neurons", 10, 1000, step=10)
        lambd = trial.suggest_float("lambda", 0.0, 0.0, step=0.01)
        data = run_NEF_rec("carrabin", sid, alpha=alpha, z=0, lambd=lambd, n_neurons=n_neurons)
    loss = QID_loss([], model_type, sid)
    return loss

def NEF_jiang_loss(trial, model_type, sid):
    if model_type=='NEF_RL':
        # n_all = trial.suggest_int("n_all", 20, 400, step=20)
        n_all = trial.suggest_int("n_all", 100, 100, step=20)
        alpha = trial.suggest_float("alpha", 0.01, 1.5, step=0.01)
        lambd = trial.suggest_float("lambda", 0.0, 0.0, step=0.01)
        z = trial.suggest_float("z", 0.01, 1.0, step=0.01)
        beta = trial.suggest_float("beta", 0.01, 10, step=0.01)
        data = run_RL("jiang", sid, alpha=alpha, z=z, lambd=lambd, n_neurons=n_all, n_learning=n_all, n_error=n_all)
    if model_type=='NEF_WM':
        # n_all = trial.suggest_int("n_all", 20, 400, step=20)
        n_all = trial.suggest_int("n_all", 100, 100, step=20)
        alpha = trial.suggest_float("alpha", 0.01, 1.5, step=0.01)
        lambd = trial.suggest_float("lambda", 0.0, 0.0, step=0.01)
        z = trial.suggest_float("z", 0.01, 1.0, step=0.01)
        beta = trial.suggest_float("beta", 0.01, 10, step=0.01)
        data = run_WM("jiang", sid, alpha=alpha, z=z, lambd=lambd, n_memory=n_all, n_neurons=n_all, n_error=n_all)
    if model_type=='NEF_syn':
        n_neurons = trial.suggest_int("n_neurons", 500, 500, step=20)
        alpha = trial.suggest_float("alpha", 1e-5, 1e-3, step=1e-5)
        lambd = trial.suggest_float("lambda", 0.0, 0.0, step=0.01)
        z = trial.suggest_float("z", 0.01, 1.0, step=0.01)
        beta = trial.suggest_float("beta", 0.01, 10, step=0.01)
        data = run_NEF_syn("jiang", sid, alpha=alpha, z=z, lambd=lambd, n_neurons=n_neurons)
    if model_type=='NEF_rec':
        n_neurons = trial.suggest_int("n_neurons", 500, 500, step=20)
        alpha = trial.suggest_float("alpha", 0.01, 1.5, step=0.01)
        lambd = trial.suggest_float("lambda", 0.0, 0.0, step=0.01)
        z = trial.suggest_float("z", 0.01, 1.0, step=0.01)
        beta = trial.suggest_float("beta", 0.01, 10, step=0.01)
        data = run_NEF_rec("jiang", sid, alpha=alpha, z=z, lambd=lambd, n_neurons=n_neurons)
    NLL = NLL_loss([beta], model_type, sid)
    return NLL

def math_carrabin_loss(trial, model_type, sid):
    if model_type=='RL_n':
        alpha = trial.suggest_float("alpha", 0.001, 1.0, step=0.001)
        sigma = trial.suggest_float("sigma", 0.001, 1.0, step=0.001)
        params = [alpha, sigma]
    if model_type=='B_n':
        sigma = trial.suggest_float("sigma", 0.001, 1.0, step=0.001)
        params = [sigma]
    if model_type=='DG_n':
        sigma = trial.suggest_float("sigma", 0.001, 1.0, step=0.001)
        params = [sigma]
    if model_type=='RL_nl':
        alpha = trial.suggest_float("alpha", 0.001, 1.0, step=0.001)
        sigma = trial.suggest_float("sigma", 0.001, 1.0, step=0.001)
        lambd = trial.suggest_float("lambda", 0.001, 3.0, step=0.001)
        params = [alpha, sigma, lambd]
    loss = QID_loss(params, model_type, sid)
    return loss

def math_jiang_loss(trial, model_type, sid):
    if model_type=='RL_z':
        alpha = trial.suggest_float("alpha", 0.01, 1.0, step=0.01)
        z = trial.suggest_float("z", 0.01, 1.0, step=0.01)
        beta = trial.suggest_float("beta", 0.01, 10.0, step=0.01)
        params = [alpha, z, beta]
    if model_type=='DG_z':
        z = trial.suggest_float("z", 0.01, 1.0, step=0.01)
        beta = trial.suggest_float("beta", 0.01, 10.0, step=0.01)
        params = [z, beta]
    if model_type=='RL_zl':
        alpha = trial.suggest_float("alpha", 0.01, 1.0, step=0.01)
        z = trial.suggest_float("z", 0.0, 1.0, step=0.01)
        lambd = trial.suggest_float("lambda", 0.0, 2.0, step=0.01)
        beta = trial.suggest_float("beta", 0.01, 10.0, step=0.01)
        params = [alpha, z, lambd, beta]
    loss = NLL_loss(params, model_type, sid)
    return loss

def get_expectations_carrabin(model_type, params, sid, trial, stage, rng=np.random.RandomState(seed=0)):
    human = pd.read_pickle(f"data/carrabin.pkl").query("sid==@sid")
    if model_type in ['NEF_WM', 'NEF_RL', 'NEF_syn', 'NEF_rec']:
        nef_data = pd.read_pickle(f"data/{model_type}_carrabin_{sid}_estimates.pkl")
        expectation = nef_data.query("trial==@trial & stage==@stage")['estimate'].unique()[0]
    else:
        subdata = human.query("trial==@trial & stage<=@stage")
        colors = subdata['color'].to_numpy()
        expectation = 0
        for c, color in enumerate(colors):
            error = color - expectation
            if model_type == 'B_n':
                weight = 1 / (c+3)
                eps = rng.normal(0, params[0])
            if model_type == 'DG_n':
                weight = 1 / (c+1)
                eps = rng.normal(0, params[0])
            if model_type == 'RL_n':
                weight = params[0]
                eps = rng.normal(0, params[1])
            if model_type == 'RL_nl':
                weight = params[0] * np.power(c+1, -params[2])
                eps = rng.normal(0, params[1])
            expectation += weight*error + eps
            expectation = np.clip(expectation, -1, 1)
    return expectation

def get_expectations_jiang(model_type, params, sid, trial, stage, full=False):
    human = pd.read_pickle(f"data/jiang.pkl").query("sid==@sid")
    if model_type in ['NEF_WM', 'NEF_RL', 'NEF_syn', 'NEF_rec']:
        nef_data = pd.read_pickle(f"data/{model_type}_jiang_{sid}_estimates.pkl")
        expectation = nef_data.query("trial==@trial & stage==@stage")['estimate'].unique()[0]
    else:
        subdata = human.query("trial==@trial & stage<=@stage")
        colors = subdata['color'].to_numpy()
        RDs = subdata['RD'].to_numpy()
        expectation = 0
        expectations = [0]
        for c, color in enumerate(colors):
            stg = int(subdata.iloc[c]['stage'])
            error = color - expectation
            RD = 0 if stg in [0,1] else RDs[c]
            if model_type == 'DG_z':
                z = params[0]
                weight = 1 / (c+1) + z*RD
            if model_type == 'RL_z':
                z = params[1]
                weight = 1 if stg==0 else params[0] + z*RD
            if model_type == 'RL_zl':
                z = params[0]
                weight = 1 if stg==0 else params[0] * np.power(c+1, -params[1]) + z*RD
            weight = np.clip(weight, 0, 1)
            expectation += weight * error
            expectation = np.clip(expectation, -1, 1)
            expectations.append(expectation)
    return expectation if not full else expectations

def mean_loss(params, model_type, sid):
    human = pd.read_pickle(f"data/carrabin.pkl").query("sid==@sid")
    trials = human['trial'].unique()
    stages = human['stage'].unique()
    errors = []
    for trial in trials:
        for stage in stages:
            response_model = get_expectations_carrabin(model_type, params, sid, trial, stage, rng=np.random.RandomState(seed=sid+1000*trial))
            response_human = human.query("trial==@trial and stage==@stage")['response'].unique()[0]
            errors.append(np.abs(response_human - response_model))
    error = np.mean(errors)
    return error

def QID_loss(params, model_type, sid):  # Carrabin loss function based on distribution of responses to each input sequence
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

def NLL_loss(params, model_type, sid):
    NLL = 0
    human = pd.read_pickle(f"data/jiang.pkl").query("sid==@sid")
    trials = human['trial'].unique()
    stages = human['stage'].unique()
    beta = params[-1]
    for trial in trials:
        for stage in stages:
            expectation = get_expectations_jiang(model_type, params, sid, trial, stage)
            act = human.query("trial==@trial and stage==@stage")['action'].unique()[0]
            prob = scipy.special.expit(beta*expectation)
            NLL -= np.log(prob) if act==1 else np.log(1-prob)
    return NLL

def fit_carrabin(model_type, sid, method, optuna_trials=1):
    if method=='optuna':
        study = optuna.create_study(direction="minimize")
        if model_type in ['NEF_RL', 'NEF_WM', 'NEF_syn', 'NEF_rec']:
            study.optimize(lambda trial: NEF_carrabin_loss(trial, model_type, sid), n_trials=optuna_trials)
        else:
            study.optimize(lambda trial: math_carrabin_loss(trial, model_type, sid), n_trials=optuna_trials)
        best_params = study.best_trial.params
        loss = study.best_trial.value
        param_names = ["type", "sid"]
        params = [model_type, sid]
        for key, value in best_params.items():
            param_names.append(key)
            params.append(value)
        if model_type == 'NEF_RL':
            alpha = best_params['alpha']
            lambd = best_params['lambda']
            n_all = best_params['n_all']
            data = run_RL("carrabin", sid, alpha=alpha, lambd=lambd, z=0, n_neurons=n_all, n_learning=n_all, n_error=n_all)
        if model_type == 'NEF_WM':
            alpha = best_params['alpha']
            lambd = best_params['lambda']
            n_all = best_params['n_all']
            data = run_WM("carrabin", sid, alpha=alpha, lambd=lambd, z=0, n_memory=n_all, n_neurons=n_all, n_error=n_all)
        if model_type == 'NEF_syn':
            alpha = best_params['alpha']
            lambd = best_params['lambda']
            n_neurons = best_params['n_neurons']
            data = run_WM("carrabin", sid, alpha=alpha, lambd=lambd, z=0, n_neurons=n_neurons)
        error = mean_loss(params[2:], model_type, sid)
        print(f"{len(study.trials)} trials completed. Best value is {loss:.4} with error {error:.4}:")
    performance_data = pd.DataFrame([[model_type, sid, loss, error]], columns=['type', 'sid', 'loss', 'error'])
    performance_data.to_pickle(f"data/{model_type}_{dataset}_{sid}_performance.pkl")
    fitted_params = pd.DataFrame([params], columns=param_names)
    fitted_params.to_pickle(f"data/{model_type}_{dataset}_{sid}_params.pkl")
    return performance_data, fitted_params

def fit_jiang(model_type, sid, method, optuna_trials=200):
    if method=='optuna':
        study = optuna.create_study(direction="minimize")
        if model_type in ['NEF_RL', 'NEF_WM']:
            study.optimize(lambda trial: NEF_jiang_loss(trial, model_type, sid), n_trials=optuna_trials)
        else:
            study.optimize(lambda trial: math_jiang_loss(trial, model_type, sid), n_trials=optuna_trials)
        best_params = study.best_trial.params
        loss = study.best_trial.value
        param_names = ["type", "sid"]
        params = [model_type, sid]
        for key, value in best_params.items():
            param_names.append(key)
            params.append(value)
        print(f"{len(study.trials)} trials completed. Best value is {loss:.4} with parameters:")
    performance_data = pd.DataFrame([[model_type, sid, loss]], columns=['type', 'sid', 'loss'])
    performance_data.to_pickle(f"data/{model_type}_{dataset}_{sid}_performance.pkl")
    fitted_params = pd.DataFrame([params], columns=param_names)
    fitted_params.to_pickle(f"data/{model_type}_{dataset}_{sid}_params.pkl")
    return performance_data, fitted_params

if __name__ == '__main__':
    dataset = sys.argv[1]
    model_type = sys.argv[2]
    sid = int(sys.argv[3])
    method = 'optuna'
    start = time.time()
    print(f"fitting {model_type}, {sid}")
    if dataset=='carrabin':
        performance_data, fitted_params = fit_carrabin(model_type, sid, method)
    elif dataset=='jiang':
        performance_data, fitted_params = fit_jiang(model_type, sid, method)
    print(performance_data)
    print(fitted_params)
    end = time.time()
    print(f"runtime {(end-start)/60:.4} min")