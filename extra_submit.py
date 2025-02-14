import sys
import pandas as pd
import subprocess
import time


if experiment=='noise_RL':
	experiment = sys.argv[1]
	n_neurons = [int(arg) for arg in sys.argv[2:]]
	sids = pd.read_pickle("data/carrabin.pkl")['sid'].unique()
	n = 0
	for sid in sids:
		for n1 in n_neurons:
			for n2 in n_neurons:
				n += 1
				submit_string = ["sbatch", f"extra_{n}.sh"]
				a = subprocess.run(submit_string)
				time.sleep(10)  # wait a few seconds before next submission to help out SLURM system

if experiment=='noise_WM':
	experiment = sys.argv[1]
	sid = int(sys.argv[2])
	n_neurons = [int(arg) for arg in sys.argv[3:]]
	n = 0
	for n1 in n_neurons:
		for n2 in n_neurons:
			n += 1
			submit_string = ["sbatch", f"extra_{n}.sh"]
			a = subprocess.run(submit_string)
			time.sleep(10)  # wait a few seconds before next submission to help out SLURM system