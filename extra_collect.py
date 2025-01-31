import sys
import pandas as pd
import subprocess
import time

experiment = sys.argv[1]
sid = int(sys.argv[2])
label = sys.argv[3]
n_neurons = [int(arg) for arg in sys.argv[4:]]

dfs = []
if experiment=='variance_LR':
	for n in n_neurons:
		dfs.append(pd.read_pickle(f"data/NEF_RL_variance_LR_carrabin_{sid}_{n}.pkl"))
	alpha_data = pd.concat(dfs, ignore_index=True)
	alpha_data.to_pickle(f"data/NEF_RL_variance_LR_carrabin_alpha_{label}.pkl")
if experiment=='noise_WM':
	for n1 in n_neurons:
		for n2 in n_neurons:
			dfs.append(pd.read_pickle(f"data/NEF_WM_noise_WM_carrabin_{sid}_{n1}_{n2}.pkl"))
	noise_data = pd.concat(dfs, ignore_index=True)
	noise_data.to_pickle(f"data/NEF_WM_noise_WM_carrabin_{label}.pkl")