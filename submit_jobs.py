import sys
import pandas as pd
import subprocess
import time

## Working Memory
sids = pd.read_pickle("data/behavior.pkl")['sid'].unique()
for sid in sids:
   delete_string = ["rm", f"data/wm_{sid}.pkl"]
   submit_string = ["sbatch", f"wm_{sid}.sh"]
   a = subprocess.run(delete_string)
   b = subprocess.run(submit_string)
   time.sleep(0.5)