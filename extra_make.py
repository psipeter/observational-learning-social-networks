import sys
import pandas as pd
import subprocess

experiment = sys.argv[1]

if experiment=='noise_RL':
	n_neurons = [int(arg) for arg in sys.argv[2:]]
	# n_neurons = [int(arg) for arg in sys.argv[2:-1]]
	# paramfile = sys.argv[-1]
	# paramfile = "RL_n2"
	paramfile = "NEF_RL"
	sids = pd.read_pickle("data/carrabin.pkl")['sid'].unique()
	n = 0
	for sid in sids:
		for n1 in n_neurons:
			for n2 in n_neurons:
				n += 1
				run_string = f"python noise_RL.py {sid} {n1} {n2} {paramfile}"
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
					rsh.write('''#SBATCH --time=4:00:0''')
					rsh.write("\n")
					rsh.write(run_string)

if experiment=='noise_WM':
	sid = int(sys.argv[2])
	n_neurons = [int(arg) for arg in sys.argv[3:]]
	n = 0
	for n1 in n_neurons:
		for n2 in n_neurons:
			n += 1
			run_string = f"python noise_WM.py {sid} {n1} {n2}"
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
				rsh.write('''#SBATCH --time=4:00:0''')
				rsh.write("\n")
				rsh.write(run_string)