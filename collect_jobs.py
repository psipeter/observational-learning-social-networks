import sys
import pandas as pd
import subprocess

## Working Memory
sids = pd.read_pickle("data/behavior.pkl")['sid'].unique()
dfs = []
for sid in sids:
   try:
      df = pd.read_pickle(f"data/wm_sid{sid}.pkl")
      dfs.append(df)
   except:
      print(f"sid {sid} missing")
data = pd.concat(dfs, ignore_index=True)
data.to_pickle("data/WM.pkl")