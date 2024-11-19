import sys
import pandas as pd
import subprocess

model_type = sys.argv[1]

if model_type in ['NEF-WM', 'NEF-RL', 'RL1']:
   sids = pd.read_pickle("data/behavior.pkl")['sid'].unique()
   for sid in sids:
      make_string = f"python fit.py {model_type} {sid}"
      with open (f'wm_{sid}.sh', 'w') as rsh:
         rsh.write('''#!/bin/bash''')
         rsh.write("\n")
         rsh.write('''#SBATCH --mem=8G''')
         rsh.write("\n")
         rsh.write('''#SBATCH --nodes=1''')
         rsh.write("\n")
         rsh.write('''#SBATCH --ntasks-per-node=1''')
         rsh.write("\n")
         rsh.write('''#SBATCH --time=0:10:0''')
         rsh.write("\n")
         rsh.write(make_string)

if model_type=='WM':
   z = float(sys.argv[2])
   k = float(sys.argv[3])
   sids = pd.read_pickle("data/behavior.pkl")['sid'].unique()
   for sid in sids:
      make_string = f"python run.py WM {sid} {z} {k}"
      with open (f'wm_{sid}.sh', 'w') as rsh:
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

if model_type=='RL':
   z = float(sys.argv[2])
   k = float(sys.argv[3])
   learning_rate = float(sys.argv[4])
   sids = pd.read_pickle("data/behavior.pkl")['sid'].unique()
   for sid in sids:
      make_string = f"python run.py RL {sid} {z} {k} {learning_rate}"
      with open (f'rl_{sid}.sh', 'w') as rsh:
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