import sys
import pandas as pd
import subprocess

## Working Memory
z = float(sys.argv[1])
sids = pd.read_pickle("data/behavior.pkl")['sid'].unique()
for sid in sids:
   make_string = f"python run.py WM {sid} {z}"
   with open (f'wm_{sid}.sh', 'w') as rsh:
      rsh.write('''#!/bin/bash''')
      rsh.write("\n")
      rsh.write('''#SBATCH --nodes=1''')
      rsh.write("\n")
      rsh.write('''#SBATCH --ntasks-per-node=1''')
      rsh.write("\n")
      rsh.write('''#SBATCH --time=0:30:0''')
      rsh.write("\n")
      rsh.write(make_string)