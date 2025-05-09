import sys
import pandas as pd
import subprocess

dataset = sys.argv[1]
model_type = sys.argv[2]
sids = pd.read_pickle(f"data/{dataset}.pkl")['sid'].unique()

for sid in sids:
   if model_type in ['NEF_WM', 'NEF_RL', 'NEF_syn', 'NEF_rec']:
      paramfile = model_type
      optimize_string = f"python fit.py {dataset} {model_type} {sid}"
      fit_string = f"python run.py {dataset} {model_type} {sid} {paramfile}"
      rerun_string = f"python rerun.py {dataset} {model_type} {sid}"
      file_string = f'nef_{sid}.sh'
   else:
      fit_string = f"python fit.py {dataset} {model_type} {sid}"
      rerun_string = f"python rerun.py {dataset} {model_type} {sid}"
      file_string = f'fit_{sid}.sh'
   with open (file_string, 'w') as rsh:
      rsh.write('''#!/bin/bash''')
      rsh.write("\n")
      rsh.write('''#SBATCH --mem=4G''')
      rsh.write("\n")
      rsh.write('''#SBATCH --nodes=1''')
      rsh.write("\n")
      rsh.write('''#SBATCH --ntasks-per-node=1''')
      if model_type in ['NEF_WM', 'NEF_RL', 'NEF_syn', 'NEF_rec']:
         rsh.write("\n")
         rsh.write('''#SBATCH --time=48:0:0''')
         rsh.write("\n")
         rsh.write(optimize_string)
      else:
         rsh.write("\n")
         rsh.write('''#SBATCH --time=4:0:0''')
      rsh.write("\n")
      rsh.write(fit_string)
      rsh.write("\n")
      rsh.write(rerun_string)