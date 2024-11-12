import sys
import pandas as pd
import subprocess
import time

## Working Memory
sids = pd.read_pickle("data/behavior.pkl")['sid'].unique()
for sid in sids:
   delete_string = ["rm", f"data/wm_{sid}.pkl"]
   delete_string2 = ["rm", f"slurm-*"]
   submit_string = ["sbatch", f"wm_{sid}.sh"]
   a = subprocess.run(delete_string)
   b = subprocess.run(submit_string)
   c = subprocess.run(delete_string2)
   # time.sleep(0.5)