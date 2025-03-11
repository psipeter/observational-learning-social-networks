import sys
import pandas as pd
import subprocess
import time

experiment = sys.argv[1]

if experiment=='noise_vs_neurons':
	model_type = sys.argv[2]
	sid = int(sys.argv[3])
	# alpha = sys.argv[4]
	n_neurons = [int(arg) for arg in sys.argv[4:-1]]
	label = sys.argv[-1]
	dfs = []
	for n1 in n_neurons:
		for n2 in n_neurons:
			dfs.append(pd.read_pickle(f"data/noise_vs_neurons_{model_type}_{sid}_{n1}_{n2}.pkl"))
	noise_data = pd.concat(dfs, ignore_index=True)
	noise_data.to_pickle(f"data/noise_vs_neurons_{model_type}_{sid}_{label}.pkl")

if experiment=='learning_noise':
	model_type = sys.argv[2]
	n_neurons = [int(arg) for arg in sys.argv[3:-1]]
	label = sys.argv[-1]
	sids = pd.read_pickle("data/jiang.pkl")['sid'].unique()
	dfs = []
	n = 0
	for sid in sids:
		for neurons in n_neurons:
			n += 1
			dfs.append(pd.read_pickle(f"data/{model_type}_{sid}_{neurons}_learning_noise.pkl"))
	noise_data = pd.concat(dfs, ignore_index=True)
	noise_data.to_pickle(f"data/{model_type}_{label}_learning_noise.pkl")