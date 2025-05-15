import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import nengo
import pandas as pd
import time
from NEF_rec import *
from NEF_syn import *
from environments import *
from fit import *

dataset = "yoo"
model_type = "NEF_syn"
alpha = 1e-3
z = 0

sid = int(sys.argv[1])
lambd = float(sys.argv[2])
n_neurons = int(sys.argv[3])

print(f"sid {sid}, lambda {lambd}, neurons {n_neurons}")

empirical = pd.read_pickle(f"data/{dataset}.pkl").query("sid==@sid")
trials = empirical['trial'].unique() 
columns = ['type', 'sid', 'trial', 'stage', 'lambda', 'n_neurons', 'error']
dfs = []
W = np.zeros((1, n_neurons))
for trial in trials[:20]:
    print(f"training sid {sid}, trial {trial}")
    env = EnvironmentCount(dataset, sid=sid, trial=trial, lambd=lambd)
    net, sim, W = simulate_NEF_syn(W, env, alpha=alpha, n_neurons=n_neurons, z=z, seed_net=sid, train=True)
np.savez(f"data/NEF_syn_{dataset}_{sid}_pretrained_weight.npz", W=W)
for trial in trials:
    print(f"running sid {sid}, trial {trial}")
    env = EnvironmentCount(dataset, sid=sid, trial=trial, lambd=lambd)
    net, sim = simulate_NEF_syn(W, env, alpha=alpha, n_neurons=n_neurons, z=z, seed_net=sid, train=False)
    obs_times = env.obs_times
    for s, tidx in enumerate(obs_times):
        stage = env.stages[s]
        weight = np.mean(sim.data[net.probe_weight][tidx-100: tidx])
        target = np.mean(sim.data[net.probe_target][tidx-100: tidx])
        error = np.abs(target - weight)
        dfs.append(pd.DataFrame([[model_type, sid, trial, stage, lambd, n_neurons, error]], columns=columns))
data = pd.concat(dfs, ignore_index=True)
data.to_pickle(f"data/weighted_error_{sid}_{lambd}_{n_neurons}.pkl")
print(data)
