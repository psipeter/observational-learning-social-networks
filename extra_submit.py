import sys
import pandas as pd
import subprocess
import time

experiment = sys.argv[1]

if experiment=='noise_vs_neurons':
	model_type = sys.argv[2]
	sid = int(sys.argv[3])
	# alpha = sys.argv[4]
	n_neurons = [int(arg) for arg in sys.argv[4:]]
	n = 0
	for n1 in n_neurons:
		for n2 in n_neurons:
			n += 1
			submit_string = ["sbatch", f"extra_{n}.sh"]
			a = subprocess.run(submit_string)
			time.sleep(1)

if experiment=='learning_noise':
	model_type = sys.argv[2]
	n_neurons = [int(arg) for arg in sys.argv[3:]]
	sids = pd.read_pickle("data/jiang.pkl")['sid'].unique()
	n = 0
	for sid in sids:
		for neurons in n_neurons:
			n += 1
			submit_string = ["sbatch", f"extra_{n}.sh"]
			a = subprocess.run(submit_string)
			time.sleep(1)

if experiment=='counting':
	dataset = sys.argv[2]
	n_sid = int(sys.argv[3])
	n_neurons = [int(arg) for arg in sys.argv[4:]]
	sids = pd.read_pickle(f"data/{dataset}.pkl")['sid'].unique()[:n_sid]
	n = 0
	for sid in sids:
		for neurons in n_neurons:
			n += 1
			submit_string = ["sbatch", f"extra_{n}.sh"]
			a = subprocess.run(submit_string)
			time.sleep(1)

if experiment=='activities':
	model_type = sys.argv[1]
	sids = pd.read_pickle(f"data/jiang.pkl")['sid'].unique()
	for sid in sids:
		submit_string = ["sbatch", f"extra_{sid}.sh"]
		a = subprocess.run(submit_string)
		time.sleep(1)