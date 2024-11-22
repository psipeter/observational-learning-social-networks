import sys
import pandas as pd
import subprocess

model_type = sys.argv[1]
label = sys.argv[2]
sids = pd.read_pickle("data/behavior.pkl")['sid'].unique()
dfs = []

for sid in sids:
	try:
		if model_type in ['NEF-WM', 'NEF-RL', 'RL1', 'RL3', 'RL3rd', 'ZK', 'DGn', 'DGrds']:
			df = pd.read_pickle(f"data/{model_type}_{sid}.pkl")
		if model_type in ['WM', 'RL']:
			df = pd.read_pickle(f"data/{model_type}_{sid}.pkl")
		dfs.append(df)
	except:
		print(f"sid {sid} missing")
data = pd.concat(dfs, ignore_index=True)
data.to_pickle(f"data/{model_type}_{label}.pkl")