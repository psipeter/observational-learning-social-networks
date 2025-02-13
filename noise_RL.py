import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import nengo
import scipy
import time
import sys
from NEF_RL import *

def noise_RL(sid, trial, n_learning, n_error, s):
    seed_net = sid + 1000*trial
    columns = ['type', 'sid', 'trial', 'stage', 'qid', 'n_learning', 'n_error', 'response']
    dfs = []
    env = Environment(dataset="carrabin", sid=sid, trial=trial, decay="stages", s=s)
    net, sim = simulate_RL(env=env, seed_net=seed_net, n_learning=n_learning, n_error=n_error, progress_bar=False)
    for stage in env.stages:
        tidx = int((stage*env.T)/env.dt)-2
        response = sim.data[net.probe_value][tidx][0]
        qid = pd.read_pickle(f"data/carrabin.pkl").query("sid==@sid & trial==@trial and stage==@stage")['qid'].unique()[0]
        df = pd.DataFrame([['NEF_RL', sid, trial, stage, qid, n_learning, n_error, response]], columns=columns)
        dfs.append(df)
    data = pd.concat(dfs, ignore_index=True)
    return data

if __name__ == '__main__':
    sid = int(sys.argv[1])
    n_learning = int(sys.argv[2])
    n_error = int(sys.argv[3])
    # mu = float(sys.argv[4])
    paramfile = sys.argv[4]
    params = pd.read_pickle(f"data/{paramfile}_carrabin_{sid}_params.pkl")
    mu = params['mu'].unique()[0]
    empirical = pd.read_pickle(f"data/carrabin.pkl")
    trials = empirical['trial'].unique()
    start = time.time()
    dfs = []
    for trial in trials:
        print(f"sid {sid}, trial {trial}")
        dfs.append(noise_RL(sid, trial, n_learning, n_error, s=[mu,mu,mu,mu,mu]))
    noise_data = pd.concat(dfs, ignore_index=True)
    noise_data.to_pickle(f"data/NEF_RL_noise_RL_carrabin_{sid}_{n_learning}_{n_error}.pkl")
    print(noise_data)
    end = time.time()
    print(f"runtime {(end-start)/60:.4} min")