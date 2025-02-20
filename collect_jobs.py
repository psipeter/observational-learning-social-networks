import sys
import pandas as pd
import subprocess

dataset = sys.argv[1]
model_type = sys.argv[2]
label = sys.argv[3]
sids = pd.read_pickle(f"data/{dataset}.pkl")['sid'].unique()

dfs1 = []
dfs2 = []
dfs3 = []

for sid in sids:
	try:
		df1 = pd.read_pickle(f"data/{model_type}_{dataset}_{sid}_performance.pkl")
		df2 = pd.read_pickle(f"data/{model_type}_{dataset}_{sid}_params.pkl")
		df3 = pd.read_pickle(f"data/{model_type}_{dataset}_{sid}_dynamics.pkl")
		dfs1.append(df1)
		dfs2.append(df2)
		dfs3.append(df3)
	except:
		print(f"missing {model_type} {sid}")

performance = pd.concat(dfs1, ignore_index=True)
params = pd.concat(dfs2, ignore_index=True)
reruns = pd.concat(dfs3, ignore_index=True)
performance.to_pickle(f"data/{model_type}_{dataset}_{label}_performance.pkl")
params.to_pickle(f"data/{model_type}_{dataset}_{label}_params.pkl")
reruns.to_pickle(f"data/{model_type}_{dataset}_{label}_dynamics.pkl")
