import sys
import pandas as pd
import subprocess

## Working Memory
sids = pd.read_pickle("data/behavior.pkl")['sid'].unique()
for sid in sids:
   submit_string = ["sbatch", f"WM_{sid}.sh"]
   a = subprocess.run(submit_string)