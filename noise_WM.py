import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import nengo
import scipy
import time
import sys
from NEF_WM import *

def bayes(empirical, stage):
    subdata = empirical.query("stage<=@stage")
    reds = subdata.query("color==1")['color'].size
    p_star = (reds+1)/(stage+2)
    expectation = 2*p_star-1
    return expectation

def noise_WM(sid, trial, n_number, n_working):
    empirical = pd.read_pickle(f"data/carrabin.pkl").query("sid==@sid & trial==@trial")
    seed_net = sid + 1000*trial
    columns = ['type', 'n_number', 'n_working', 'sid', 'trial', 'stage', 'error number', 'error expectation']
    dfs = []
    env = Environment(dataset="carrabin", sid=sid, trial=trial)
    net, sim = simulate_WM(env=env, n_number=n_number, n_working=n_working, seed_net=seed_net, z=0, progress_bar=False)
    for stage in env.stages:
        tidx = int((stage*env.T)/env.dt)-2
        number = sim.data[net.probe_number][tidx][0]
        expectation_NEF = sim.data[net.probe_memory][tidx][0]
        expectation_bayes = bayes(empirical, stage)
        delta_N = np.abs(number - stage) / stage
        delta_E = np.abs(expectation_NEF - expectation_bayes)
        df = pd.DataFrame([['NEF_WM', n_number, n_working, sid, trial, stage, delta_N, delta_E]], columns=columns)
        dfs.append(df)
    data = pd.concat(dfs, ignore_index=True)
    return data

if __name__ == '__main__':
    sid = int(sys.argv[1])
    n_number = int(sys.argv[2])
    n_working = int(sys.argv[3])
    empirical = pd.read_pickle(f"data/carrabin.pkl")
    trials = empirical['trial'].unique()

    start = time.time()
    dfs = []
    for trial in trials:
        print(f"sid {sid}, trial {trial}")
        dfs.append(noise_WM(sid, trial, n_number, n_working))
    noise_data = pd.concat(dfs, ignore_index=True)
    noise_data.to_pickle(f"data/NEF_WM_noise_WM_carrabin_{sid}_{n_number}_{n_working}.pkl")
    print(noise_data)
    end = time.time()
    print(f"runtime {(end-start)/60:.4} min")