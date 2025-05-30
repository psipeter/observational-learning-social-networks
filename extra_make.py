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
			run_string = f"python noise_vs_neurons.py {model_type} {sid} {n1} {n2}"
			file_string = f'extra_{n}.sh'
			with open (file_string, 'w') as rsh:
				rsh.write('''#!/bin/bash''')
				rsh.write("\n")
				rsh.write('''#SBATCH --mem=2G''')
				rsh.write("\n")
				rsh.write('''#SBATCH --nodes=1''')
				rsh.write("\n")
				rsh.write('''#SBATCH --ntasks-per-node=1''')
				rsh.write("\n")
				rsh.write('''#SBATCH --time=1:00:0''')
				rsh.write("\n")
				rsh.write(run_string)

if experiment=='learning_noise':
	model_type = sys.argv[2]
	n_neurons = [int(arg) for arg in sys.argv[3:]]
	sids = pd.read_pickle("data/jiang.pkl")['sid'].unique()
	n = 0
	for sid in sids:
		for neurons in n_neurons:
			n += 1
			run_string = f"python learning_noise.py {model_type} {sid} {neurons}"
			file_string = f'extra_{n}.sh'
			with open (file_string, 'w') as rsh:
				rsh.write('''#!/bin/bash''')
				rsh.write("\n")
				rsh.write('''#SBATCH --mem=4G''')
				rsh.write("\n")
				rsh.write('''#SBATCH --nodes=1''')
				rsh.write("\n")
				rsh.write('''#SBATCH --ntasks-per-node=1''')
				rsh.write("\n")
				rsh.write('''#SBATCH --time=1:00:0''')
				rsh.write("\n")
				rsh.write(run_string)

if experiment=='counting':
	dataset = sys.argv[2]
	n_sid = int(sys.argv[3])
	n_neurons = [int(arg) for arg in sys.argv[4:]]
	sids = pd.read_pickle(f"data/{dataset}.pkl")['sid'].unique()[:n_sid]
	n = 0
	for sid in sids:
		for neurons in n_neurons:
			n += 1
			run_string = f"python counting.py {dataset} {sid} {neurons}"
			file_string = f'extra_{n}.sh'
			with open (file_string, 'w') as rsh:
				rsh.write('''#!/bin/bash''')
				rsh.write("\n")
				rsh.write('''#SBATCH --mem=4G''')
				rsh.write("\n")
				rsh.write('''#SBATCH --nodes=1''')
				rsh.write("\n")
				rsh.write('''#SBATCH --ntasks-per-node=1''')
				rsh.write("\n")
				rsh.write('''#SBATCH --time=1:00:0''')
				rsh.write("\n")
				rsh.write(run_string)

if experiment=='activities':
	dataset = sys.argv[2]
	model_type = sys.argv[3]
	sids = pd.read_pickle(f"data/{dataset}.pkl")['sid'].unique()
	for sid in sids:
		run_string = f"python activities.py {dataset} {model_type} {sid}"
		file_string = f'extra_{sid}.sh'
		with open (file_string, 'w') as rsh:
			rsh.write('''#!/bin/bash''')
			rsh.write("\n")
			rsh.write('''#SBATCH --mem=4G''')
			rsh.write("\n")
			rsh.write('''#SBATCH --nodes=1''')
			rsh.write("\n")
			rsh.write('''#SBATCH --ntasks-per-node=1''')
			rsh.write("\n")
			rsh.write('''#SBATCH --time=1:00:0''')
			rsh.write("\n")
			rsh.write(run_string)

if experiment=='iti_decode':
	model_type = sys.argv[2]
	sids = pd.read_pickle(f"data/carrabin.pkl")['sid'].unique()
	for sid in sids:
		run_string = f"python iti_decode.py {model_type} {sid}"
		file_string = f'extra_{sid}.sh'
		with open (file_string, 'w') as rsh:
			rsh.write('''#!/bin/bash''')
			rsh.write("\n")
			rsh.write('''#SBATCH --mem=4G''')
			rsh.write("\n")
			rsh.write('''#SBATCH --nodes=1''')
			rsh.write("\n")
			rsh.write('''#SBATCH --ntasks-per-node=1''')
			rsh.write("\n")
			rsh.write('''#SBATCH --time=1:00:0''')
			rsh.write("\n")
			rsh.write(run_string)

if experiment=='iti_noise':
	model_type = sys.argv[2]
	sids = pd.read_pickle(f"data/carrabin.pkl")['sid'].unique()
	for sid in sids:
		run_string = f"python iti_noise.py {model_type} {sid}"
		file_string = f'extra_{sid}.sh'
		with open (file_string, 'w') as rsh:
			rsh.write('''#!/bin/bash''')
			rsh.write("\n")
			rsh.write('''#SBATCH --mem=4G''')
			rsh.write("\n")
			rsh.write('''#SBATCH --nodes=1''')
			rsh.write("\n")
			rsh.write('''#SBATCH --ntasks-per-node=1''')
			rsh.write("\n")
			rsh.write('''#SBATCH --time=1:00:0''')
			rsh.write("\n")
			rsh.write(run_string)

if experiment in ['weighting_error_lambd', 'weighting_error_neurons']:
	n_sid = int(sys.argv[2])
	if experiment=='weighting_error_lambd':
		n_neurons = [int(sys.argv[3])]
		lambdas = [float(arg) for arg in sys.argv[4:]]
	if experiment=='weighting_error_neurons':
		lambdas = [float(sys.argv[3])]
		n_neurons = [int(arg) for arg in sys.argv[4:]]
	sids = pd.read_pickle(f"data/yoo.pkl")['sid'].unique()[:n_sid]
	n = 0
	for sid in sids:
		for neurons in n_neurons:
			for lambd in lambdas:
				n += 1
				run_string = f"python weighting_error.py {sid} {lambd} {neurons}"
				file_string = f'extra_{n}.sh'
				with open (file_string, 'w') as rsh:
					rsh.write('''#!/bin/bash''')
					rsh.write("\n")
					rsh.write('''#SBATCH --mem=4G''')
					rsh.write("\n")
					rsh.write('''#SBATCH --nodes=1''')
					rsh.write("\n")
					rsh.write('''#SBATCH --ntasks-per-node=1''')
					rsh.write("\n")
					rsh.write('''#SBATCH --time=1:00:0''')
					rsh.write("\n")
					rsh.write(run_string)