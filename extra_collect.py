import sys
import pandas as pd
import subprocess
import time


dfs = []
if experiment=='noise_RL':
	experiment = sys.argv[1]
	n_neurons = [int(arg) for arg in sys.argv[2:-1]]
	sids = pd.read_pickle("data/carrabin.pkl")['sid'].unique()
	label = sys.argv[-1]
	for sid in sids:
		for n1 in n_neurons:
			for n2 in n_neurons:
				dfs.append(pd.read_pickle(f"data/NEF_RL_noise_RL_carrabin_{sid}_{n1}_{n2}.pkl"))
	noise_data = pd.concat(dfs, ignore_index=True)
	noise_data.to_pickle(f"data/NEF_RL_noise_RL_carrabin_{label}.pkl")
if experiment=='noise_WM':
	experiment = sys.argv[1]
	sid = int(sys.argv[2])
	n_neurons = [int(arg) for arg in sys.argv[3:-1]]
	label = sys.argv[-1]
	for n1 in n_neurons:
		for n2 in n_neurons:
			dfs.append(pd.read_pickle(f"data/NEF_WM_noise_WM_carrabin_{sid}_{n1}_{n2}.pkl"))
	noise_data = pd.concat(dfs, ignore_index=True)
	noise_data.to_pickle(f"data/NEF_WM_noise_WM_carrabin_{label}.pkl")