import sys
import pandas as pd
import subprocess

## Working Memory
label = sys.argv[1]
sids = pd.read_pickle("data/behavior.pkl")['sid'].unique()
dfs = []
for sid in sids:
   try:
      df = pd.read_pickle(f"data/wm_{sid}.pkl")
      dfs.append(df)
   except:
      print(f"sid {sid} missing")
data = pd.concat(dfs, ignore_index=True)
data.to_pickle(f"data/WM_{label}.pkl")