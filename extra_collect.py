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
		dfs.append(pd.read_pickle(f"data/NEF_RL_variance_LR_carrabin_{sid}_{n_neurons}.pkl"))
	alpha_data = pd.concat(dfs, ignore_index=True)
	alpha_data.to_pickle(f"data/NEF_RL_variance_LR_carrabin_alpha_{label}.pkl")
