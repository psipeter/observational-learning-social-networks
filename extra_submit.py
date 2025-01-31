import sys
import pandas as pd
import subprocess
import time

experiment = sys.argv[1]
sid = int(sys.argv[2])
n_neurons = [int(arg) for arg in sys.argv[3:]]

if experiment=='variance_LR':
	for n in n_neurons:
		submit_string = ["sbatch", f"extra_{n}.sh"]
		a = subprocess.run(submit_string)
		time.sleep(10)  # wait a few seconds before next submission to help out SLURM system