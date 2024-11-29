import sys
import pandas as pd
import subprocess
import time

model_type = sys.argv[1]
sids = pd.read_pickle("data/behavior.pkl")['sid'].unique()
c = subprocess.run("rm *.out", shell=True)

for sid in sids:
	if model_type in ['all', 'NEF-WM', 'NEF-RL', 'RL1', 'RL3', 'RL3rd', 'ZK', 'DGn', 'DGrd', 'DGrds', 'DGrdp']:
		# delete_string = f"rm data/*_{sid}_performance.pkl data/*_{sid}_params.pkl"
		submit_string = ["sbatch", f"fit_{sid}.sh"]
	if model_type in ['WM', 'RL']:
		delete_string = f"rm data/{model_type}_{sid}.pkl"
		submit_string = ["sbatch", f"nef_{sid}.sh"]
	# a = subprocess.run(delete_string, shell=True)
	b = subprocess.run(submit_string)
	time.sleep(10)  # wait 10s before next submission to help out SLURM system