import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import nengo
import scipy
import time
import sys
from fit import *
from NEF_RL import *
from NEF_WM import *

if __name__ == '__main__':
    model_type = sys.argv[1]
    sid = int(sys.argv[2])
    neurons = int(sys.argv[3])
    empirical = pd.read_pickle(f"data/jiang.pkl").query("sid==@sid")
    trials = empirical['trial'].unique()
    start = time.time()
    if model_type=='NEF_RL':
        p_nef = pd.read_pickle("data/NEF_RL_jiang_mar11_params.pkl").query("sid==@sid")
        alpha = p_nef['alpha'].unique()[0]
        z = p_nef['z'].unique()[0]
        lambd = p_nef['lambda'].unique()[0]
        beta_nef = p_nef['beta'].unique()[0]
        params_nef = [beta_nef]
        nef_data = run_RL("jiang", sid, alpha=alpha, z=z, lambd=lambd, n_neurons=neurons, n_learning=neurons, n_error=neurons)
        math_type = "RL_z"
        p_math = pd.read_pickle(f"data/{math_type}_jiang_mar7_params.pkl").query("sid==@sid")
        alpha_math = p_math['alpha'].unique()[0]
        z_math = p_math['z'].unique()[0]
        beta_math = p_math['beta'].unique()[0]
        params_math = [alpha_math, z_math, beta_math]
    rng = np.random.RandomState(seed=0)
    if model_type=='NEF_WM':
        p_nef = pd.read_pickle("data/NEF_WM_jiang_mar11_params.pkl").query("sid==@sid")
        alpha = p_nef['alpha'].unique()[0]
        z = p_nef['z'].unique()[0]
        lambd = p_nef['lambda'].unique()[0]
        beta_nef = p_nef['beta'].unique()[0]
        params_nef = [beta_nef]
        nef_data = run_WM("jiang", sid, alpha=alpha, z=z, lambd=lambd, n_neurons=neurons, n_memory=neurons, n_error=neurons)
        math_type = "DG_z"
        p_math = pd.read_pickle(f"data/{math_type}_jiang_mar7_params.pkl").query("sid==@sid")
        z_math = p_math['z'].unique()[0]
        beta_math = p_math['beta'].unique()[0]
        params_math = [z_math, beta_math]
    human = pd.read_pickle("data/jiang.pkl").query("sid==@sid")
    trials = human.query("sid==@sid")['trial'].unique()
    columns = ['type', 'neurons', 'sid', 'trial', 'stage', 'is_greedy', 'noise_driven']
    dfs = []
    for trial in trials:
        stages = human.query("sid==@sid & trial==@trial")['stage'].unique()
        for stage in stages:
            expectation_math = get_expectations_jiang(math_type, params_math, sid, trial, stage)
            # expectation_nef = get_expectations_jiang(model_type, params_nef, sid, trial, stage)
            expectation_nef = nef_data.query("trial==@trial & stage==@stage")['estimate'].unique()[0]
            prob_math = scipy.special.expit(beta_math*expectation_math)
            prob_nef = scipy.special.expit(beta_nef*expectation_nef)
            action_math = 1 if rng.uniform(0,1) < prob_math else -1
            action_nef = 1 if rng.uniform(0,1) < prob_nef else -1
            action_human = human.query("trial==@trial & stage==@stage")['action'].unique()[0]
            sign_math = 1 if expectation_math > 0 else -1
            sign_nef = 1 if expectation_nef > 0 else -1
            is_greedy = True if action_human==sign_math else False
            noise_driven = True if sign_nef!=sign_math else False
            dfs.append(pd.DataFrame([[model_type, neurons, sid, trial, stage, is_greedy, noise_driven]], columns=columns))
    noise_data = pd.concat(dfs, ignore_index=True)
    noise_data.to_pickle(f"data/{model_type}_{sid}_{neurons}_learning_noise.pkl")
    end = time.time()
    print(f"runtime {(end-start)/60:.4} min")