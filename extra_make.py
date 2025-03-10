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