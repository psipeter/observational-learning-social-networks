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
	n = 0
	for n1 in n_neurons:
		for n2 in n_neurons:
			n += 1
			submit_string = ["sbatch", f"extra_{n}.sh"]
			a = subprocess.run(submit_string)
			time.sleep(1)