import sys
import pandas as pd

z = float(sys.argv[1])
sids = pd.read_pickle("data/behavior.pkl")['sid'].unique()

for sid in sids:
   string = f"python run.py WM {sid} {z}"
   with open (f'WM_{sid}.sh', 'w') as rsh:
      rsh.write('''#!/bin/bash''')
      rsh.write("\n")
      rsh.write('''#SBATCH --nodes=1''')
      rsh.write("\n")
      rsh.write('''#SBATCH --ntasks-per-node=1''')
      rsh.write("\n")
      rsh.write('''#SBATCH --time=0:20:0''')
      rsh.write("\n")
      rsh.write(string)