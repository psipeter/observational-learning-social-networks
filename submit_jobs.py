import sys
import pandas as pd
import subprocess
import time

dataset = sys.argv[1]
model_type = sys.argv[2]
if dataset=='carrabin':
   sids = pd.read_pickle("data/carrabin.pkl")['sid'].unique()
elif dataset=='jiang':
   sids = pd.read_pickle("data/jiang.pkl")['sid'].unique()
# c = subprocess.run("rm *.out", shell=True)

for sid in sids:
	if model_type in ['NEF_WM', 'NEF_RL']:
		delete_string = f"rm data/{model_type}_{dataset}_{sid}.pkl"
		submit_string = ["sbatch", f"nef_{sid}.sh"]
	else:
		# delete_string = f"rm data/*_{sid}_performance.pkl data/*_{sid}_params.pkl"
		submit_string = ["sbatch", f"fit_{sid}.sh"]
	# a = subprocess.run(delete_string)
	b = subprocess.run(submit_string)
	time.sleep(10)  # wait a few seconds before next submission to help out SLURM system