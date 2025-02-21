import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import nengo
import scipy
import time
import sys
import pandas as pd
from NEF_RL import simulate_RL, EnvironmentRL
from NEF_WM2 import simulate_WM, EnvironmentWM


model_type = sys.argv[1]
sid = int(sys.argv[2])
# alpha = float(sys.argv[3])
n_other = int(sys.argv[3])
n_error = int(sys.argv[4])
# paramfile = sys.argv[3]
params = pd.read_pickle(f"data/{model_type}_carrabin_feb21_params.pkl").query("sid==@sid")
alpha = params['alpha'].unique()[0]
trials = pd.read_pickle(f"data/carrabin.pkl").query("sid==@sid")['trial'].unique()
start = time.time()

dfs = []
columns = ['type', 'sid', 'trial', 'stage', 'qid', 'n_other', 'n_error', 'response']
for trial in trials:
    print(f"sid {sid}, trial {trial}")
    seed_net = sid + 1000*trial
    if model_type=="NEF_RL":
        env = EnvironmentRL(dataset="carrabin", sid=sid, trial=trial)
        net, sim = simulate_RL(env=env, alpha=alpha, n_learning=n_other, n_error=n_error, seed_net=seed_net, progress_bar=False)
    if model_type=="NEF_WM":
        env = EnvironmentWM(dataset="carrabin", sid=sid, trial=trial)
        net, sim = simulate_WM(env=env, alpha=alpha, n_memory=n_other, n_error=n_error, seed_net=seed_net, progress_bar=False)
    for stage in env.stages:
        tidx = int((stage*env.T)/env.dt)-2
        response = np.mean(sim.data[net.probe_value][tidx-100: tidx])
        qid = pd.read_pickle(f"data/carrabin.pkl").query("sid==@sid & trial==@trial and stage==@stage")['qid'].unique()[0]
        df = pd.DataFrame([[model_type, sid, trial, stage, qid, n_other, n_error, response]], columns=columns)
        dfs.append(df)

noise_data = pd.concat(dfs, ignore_index=True)
noise_data.to_pickle(f"data/noise_vs_neurons_{model_type}_{sid}_{n_other}_{n_error}.pkl")
print(noise_data)
end = time.time()
print(f"runtime {(end-start)/60:.4} min")