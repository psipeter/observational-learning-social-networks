import sys
import pandas as pd
import subprocess
import time

model_type = sys.argv[1]

sids = pd.read_pickle("data/behavior.pkl")['sid'].unique()
c = subprocess.run("rm *.out", shell=True)

for sid in sids:
   ## Working Memory
   if model_type=='WM':
      delete_string = ["rm", f"data/wm_{sid}.pkl"]
      submit_string = ["sbatch", f"wm_{sid}.sh"]
   if model_type=='RL':
      delete_string = ["rm", f"data/rl_{sid}.pkl"]
      submit_string = ["sbatch", f"rl_{sid}.sh"]
   a = subprocess.run(delete_string)
   b = subprocess.run(submit_string)