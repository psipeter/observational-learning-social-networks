import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import nengo
import scipy
import time
import sys
from NEF_RL import *

def variance_LR(sid, trial, alpha, n_neurons, a=6e-5):
    s = [alpha,alpha,alpha,alpha,alpha]
    seed_net = sid + 1000*trial
    columns = ['type', 'n_neurons', 'sid', 'trial', 'stage', 'alpha', 'measured alpha']
    dfs = []
    env = Environment(dataset='carrabin', sid=sid, trial=trial, decay='stages', s=s)
    net, sim = simulate_RL(env=env, n_neurons=n_neurons, seed_net=seed_net, z=0, a=a, progress_bar=False)
    for stage in env.stages:
        told = int(((stage-1)*env.T)/env.dt)+2
        tnew = int((stage*env.T)/env.dt)-2
        eold = sim.data[net.probe_prediction][told][0]
        enew = sim.data[net.probe_prediction][tnew][0]
        delta_E = np.abs(enew - eold)
        color = env.empirical.query("stage==@stage")['color'].unique()[0]
        PE = np.abs(color - eold)
        measured_alpha = delta_E / PE
        # print(stage, color, eold, enew, PE, measured_alpha) 
        df = pd.DataFrame([['NEF_RL', n_neurons, sid, trial, stage, alpha, measured_alpha]], columns=columns)
        dfs.append(df)
    data = pd.concat(dfs, ignore_index=True)
    return data

if __name__ == '__main__':
	sid = int(sys.argv[1])
	n_neurons = int(sys.argv[2])
	empirical = pd.read_pickle(f"data/carrabin.pkl")
	trials = empirical['trial'].unique()
	alpha = pd.read_pickle(f"data/RL_carrabin_{sid}_params.pkl")['alpha'].unique()[0]

	start = time.time()
	dfs = []
	for trial in trials:
		print(f"sid {sid}, trial {trial}")
		dfs.append(variance_LR(sid, trial, alpha=alpha, n_neurons=n_neurons))
	alpha_data = pd.concat(dfs, ignore_index=True)
	alpha_data.to_pickle(f"data/NEF_RL_variance_LR_carrabin_{sid}_{n_neurons}.pkl")
	print(alpha_data)
	end = time.time()
	print(f"runtime {(end-start)/60:.4} min")