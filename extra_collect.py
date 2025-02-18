import sys
import pandas as pd
import subprocess
import time

experiment = sys.argv[1]

if experiment=='noise_vs_neurons':
	model_type = sys.argv[2]
	sid = int(sys.argv[3])
	alpha = sys.argv[4]
	n_neurons = [int(arg) for arg in sys.argv[5:]]
	label = sys.argv[-1]
	dfs = []
	for n1 in n_neurons:
		for n2 in n_neurons:
			dfs.append(pd.read_pickle(f"data/noise_vs_neurons_{model_type}_{sid}_{n1}_{n2}.pkl"))
	noise_data = pd.concat(dfs, ignore_index=True)
	noise_data.to_pickle(f"data/noise_vs_neurons_{model_type}_{sid}_{label}.pkl")