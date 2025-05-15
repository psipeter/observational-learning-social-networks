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
			try:
				dfs.append(pd.read_pickle(f"data/{model_type}_{sid}_{neurons}_learning_noise.pkl"))
			except:
				print(f"missing sid {sid} neurons {neurons}")
	noise_data = pd.concat(dfs, ignore_index=True)
	noise_data.to_pickle(f"data/{model_type}_{label}_learning_noise.pkl")

if experiment=='counting':
	dataset = sys.argv[2]
	n_sid = int(sys.argv[3])
	n_neurons = [int(arg) for arg in sys.argv[4:-1]]
	label = sys.argv[-1]
	sids = pd.read_pickle(f"data/{dataset}.pkl")['sid'].unique()[:n_sid]
	dfs = []
	n = 0
	for sid in sids:
		for neurons in n_neurons:
			n += 1
			try:
				dfs.append(pd.read_pickle(f"data/{dataset}_{sid}_{neurons}_counting.pkl"))
			except:
				print(f"missing sid {sid} neurons {neurons}")
	noise_data = pd.concat(dfs, ignore_index=True)
	noise_data.to_pickle(f"data/{dataset}_{label}_counting.pkl")

if experiment=='activities':
	dataset = sys.argv[2]
	model_type = sys.argv[3]
	sids = pd.read_pickle(f"data/{dataset}.pkl")['sid'].unique()
	label = sys.argv[-1]
	dfs = []
	for sid in sids:
		try:
			# dfs.append(pd.read_pickle(f"data/{model_type}_{sid}_activities.pkl"))
			dfs.append(pd.read_pickle(f"data/{dataset}_{model_type}_{sid}_activities.pkl"))
		except:
			print(f"missing sid {sid}")
	activity_data = pd.concat(dfs, ignore_index=True)
	activity_data.to_pickle(f"data/{dataset}_{model_type}_{label}_activities.pkl")

if experiment=='iti_decode':
	model_type = sys.argv[2]
	sids = pd.read_pickle(f"data/carrabin.pkl")['sid'].unique()
	label = sys.argv[-1]
	dfs = []
	for sid in sids:
		try:
			dfs.append(pd.read_pickle(f"data/{model_type}_{sid}_iti_decode.pkl"))
		except:
			print(f"missing sid {sid}")
	activity_data = pd.concat(dfs, ignore_index=True)
	activity_data.to_pickle(f"data/{model_type}_{label}_iti_decode.pkl")

if experiment=='iti_noise':
	model_type = sys.argv[2]
	sids = pd.read_pickle(f"data/carrabin.pkl")['sid'].unique()
	label = sys.argv[-1]
	dfs = []
	for sid in sids:
		try:
			dfs.append(pd.read_pickle(f"data/{model_type}_{sid}_iti_noise.pkl"))
		except:
			print(f"missing sid {sid}")
	activity_data = pd.concat(dfs, ignore_index=True)
	activity_data.to_pickle(f"data/{model_type}_{label}_iti_noise.pkl")

if experiment in ['weighting_error_lambd', 'weighting_error_neurons']:
	n_sid = int(sys.argv[2])
	if experiment=='weighting_error_lambd':
		n_neurons = [int(sys.argv[3])]
		lambdas = [float(arg) for arg in sys.argv[4:-1]]
	if experiment=='weighting_error_neurons':
		lambdas = [float(sys.argv[3])]
		n_neurons = [int(arg) for arg in sys.argv[4:-1]]
	sids = pd.read_pickle(f"data/yoo.pkl")['sid'].unique()[:n_sid]
	label = sys.argv[-1]
	dfs = []
	n = 0
	for sid in sids:
		for neurons in n_neurons:
			for lambd in lambdas:
				n += 1
				try:
					dfs.append(pd.read_pickle(f"data/weighting_error_{sid}_{lambd}_{neurons}.pkl"))
				except:
					print(f"missing sid {sid} neurons {neurons} lambda {lambd}")
	noise_data = pd.concat(dfs, ignore_index=True)
	noise_data.to_pickle(f"data/{experiment}_{label}.pkl")