import sys
import pandas as pd
import subprocess

model_type = sys.argv[1]
sids = pd.read_pickle("data/behavior.pkl")['sid'].unique()

for sid in sids:
   if model_type in ['NEF_WM']:
      paramfile = sys.argv[2]
      params = pd.read_pickle(f"data/{paramfile}_{sid}_params.pkl")
      z = params['z'].unique()[0] if sys.argv[3]=='load' else float(sys.argv[3])
      inv_temp = params['inv_temp'].unique()[0] if sys.argv[4]=='load' else float(sys.argv[4])
      fit_string = f"python run.py {model_type} {sid} {paramfile} {z} {inv_temp}"
      rerun_string = f"python rerun.py {model_type} {sid}"
      file_string = f'nef_{sid}.sh'
   elif model_type in ['NEF_RL']:
      paramfile = sys.argv[2]
      params = pd.read_pickle(f"data/{paramfile}_{sid}_params.pkl")
      z = params['z'].unique()[0] if sys.argv[3]=='load' else float(sys.argv[3])
      b = params['b'].unique()[0] if sys.argv[4]=='load' else float(sys.argv[4])
      inv_temp = params['inv_temp'].unique()[0] if sys.argv[5]=='load' else float(sys.argv[5])
      fit_string = f"python run.py {model_type} {sid} {paramfile} {z} {b} {inv_temp}"
      rerun_string = f"python rerun.py {model_type} {sid}"
      file_string = f'nef_{sid}.sh'
   else:
      fit_string = f"python fit.py {model_type} {sid}"
      rerun_string = f"python rerun.py {model_type} {sid}"
      file_string = f'fit_{sid}.sh'
   with open (file_string, 'w') as rsh:
      rsh.write('''#!/bin/bash''')
      rsh.write("\n")
      rsh.write('''#SBATCH --mem=2G''')
      rsh.write("\n")
      rsh.write('''#SBATCH --nodes=1''')
      rsh.write("\n")
      rsh.write('''#SBATCH --ntasks-per-node=1''')
      rsh.write("\n")
      rsh.write('''#SBATCH --time=1:00:0''')
      rsh.write("\n")
      rsh.write(fit_string)
      rsh.write("\n")
      rsh.write(rerun_string)