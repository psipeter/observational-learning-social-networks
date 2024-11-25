import sys
import pandas as pd
import subprocess

model_type = sys.argv[1]
label = sys.argv[2]
sids = pd.read_pickle("data/human.pkl")['sid'].unique()
dfs1 = []
dfs2 = []
dfs3 = []

for sid in sids:
	try:
		if model_type in ['NEF-WM', 'NEF-RL', 'RL1', 'RL3', 'RL3rd', 'ZK', 'DGn', 'DGrds']:
			df1 = pd.read_pickle(f"data/{model_type}_{sid}_performance.pkl")
			df2 = pd.read_pickle(f"data/{model_type}_{sid}_param.pkl")
			df3 = pd.read_pickle(f"data/{model_type}_{sid}_rerun.pkl")
			dfs1.append(df1)
			dfs2.append(df2)
			dfs3.append(df3)
		if model_type in ['WM', 'RL']:
			df1 = pd.read_pickle(f"data/{model_type}_{sid}.pkl")
			dfs1.append(df1)
	except:
		print(f"sid {sid} missing")

performance = pd.concat(dfs1, ignore_index=True)
params = pd.concat(dfs2, ignore_index=True)
reruns = pd.concat(dfs3, ignore_index=True)
performance.to_pickle(f"data/{model_type}_{label}_performance.pkl")
params.to_pickle(f"data/{model_type}_{label}_params.pkl")
reruns.to_pickle(f"data/{model_type}_{label}_reruns.pkl")