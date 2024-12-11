import sys
import pandas as pd
import subprocess

model_type = sys.argv[1]
sids = pd.read_pickle("data/behavior.pkl")['sid'].unique()

for sid in sids:
   if model_type in ['WM', 'RL']:
      z = float(sys.argv[2])
      # k = float(sys.argv[3])
      # params = pd.read_pickle(f"data/ZK_{sid}_params.pkl")  # use fitted stat model as approximate best params
      z = params['z'].unique()[0]
      # k = params['k'].unique()[0]
      # rerun_string = f"python run.py {model_type} {sid} {z} {k}"
      rerun_string = f"python run.py {model_type} {sid} {z}"
      file_string = f'nef_{sid}.sh'
   else:
      fit_string = f"python fit.py {model_type} {sid}"
      rerun_string = f"python rerun.py {model_type} {sid}"
      file_string = f'fit_{sid}.sh'
   with open (file_string, 'w') as rsh:
      rsh.write('''#!/bin/bash''')
      rsh.write("\n")
      rsh.write('''#SBATCH --mem=1G''')
      rsh.write("\n")
      rsh.write('''#SBATCH --nodes=1''')
      rsh.write("\n")
      rsh.write('''#SBATCH --ntasks-per-node=1''')
      rsh.write("\n")
      rsh.write('''#SBATCH --time=2:00:0''')
      if model_type not in ['WM', 'RL']:
         rsh.write("\n")
         rsh.write(fit_string)
      rsh.write("\n")
      rsh.write(rerun_string)