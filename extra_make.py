import sys
import pandas as pd
import subprocess

experiment = sys.argv[1]
sid = int(sys.argv[2])
n_neurons = [int(arg) for arg in sys.argv[3:]]

if experiment=='variance_LR':
   for n in n_neurons:
      run_string = f"python variance_LR.py {sid} {n}"
      file_string = f'extra_{n}.sh'
      with open (file_string, 'w') as rsh:
         rsh.write('''#!/bin/bash''')
         rsh.write("\n")
         rsh.write('''#SBATCH --mem=4G''')
         rsh.write("\n")
         rsh.write('''#SBATCH --nodes=1''')
         rsh.write("\n")
         rsh.write('''#SBATCH --ntasks-per-node=1''')
         rsh.write("\n")
         rsh.write('''#SBATCH --time=4:00:0''')
         rsh.write("\n")
         rsh.write(run_string)