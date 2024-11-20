import sys
import pandas as pd
import subprocess

model_type = sys.argv[1]
sids = pd.read_pickle("data/behavior.pkl")['sid'].unique()

if model_type in ['NEF-WM', 'NEF-RL', 'RL1', 'RL1rd', 'RL2', 'RL2rd']:
   make_string = f"python fit.py {model_type} {sid}"
   file_string = f'fit_{sid}.sh'
if model_type in ['WM', 'RL']:
   z = float(sys.argv[2])
   k = float(sys.argv[3])
   make_string = f"python run.py {model_type} {sid} {z} {k}"
   file_string = f'nef_{sid}.sh'

for sid in sids:
   with open (file_string, 'w') as rsh:
      rsh.write('''#!/bin/bash''')
      rsh.write("\n")
      rsh.write('''#SBATCH --mem=8G''')
      rsh.write("\n")
      rsh.write('''#SBATCH --nodes=1''')
      rsh.write("\n")
      rsh.write('''#SBATCH --ntasks-per-node=1''')
      rsh.write("\n")
      rsh.write('''#SBATCH --time=0:30:0''')
      rsh.write("\n")
      rsh.write(make_string)