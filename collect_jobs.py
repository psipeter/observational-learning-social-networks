import sys
import pandas as pd
import subprocess

## Working Memory
sids = pd.read_pickle("data/behavior.pkl")['sid'].unique()
for sid in sids:
   df = pd.read_pickle{f"data/WM_{sid}.pkl"}
   dfs.append(df)
data = pd.concat(dfs, ignore_index=True)
data.to_pickle("data/WM.pkl")