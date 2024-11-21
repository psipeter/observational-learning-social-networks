import sys
import pandas as pd
import subprocess

model_type = sys.argv[1]
sids = pd.read_pickle("data/behavior.pkl")['sid'].unique()
c = subprocess.run("rm *.out", shell=True)

for sid in sids:
   if model_type in ['NEF-WM', 'NEF-RL', 'RL1', 'RL1rd', 'RL2', 'RL2rd', 'ZK', 'DG']:
      delete_string = ["rm", f"data/{model_type}_{sid}.pkl", f"data/{model_type}_{sid}.npz"]
      submit_string = ["sbatch", f"fit_{sid}.sh"]
   if model_type in ['WM', 'RL']:
      delete_string = ["rm", f"data/{model_type}_{sid}.pkl"]
      submit_string = ["sbatch", f"nef_{sid}.sh"]
   a = subprocess.run(delete_string)
   b = subprocess.run(submit_string)