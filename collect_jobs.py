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
		df1 = pd.read_pickle(f"data/{model_type}_{sid}_performance.pkl")
		df2 = pd.read_pickle(f"data/{model_type}_{sid}_params.pkl")
		if model_type in ["NEF_WM", "NEF_RL"]:
			df3 = pd.read_pickle(f"data/{model_type}_{sid}_estimates.pkl")
		else:
			df3 = pd.read_pickle(f"data/{model_type}_{sid}_rerun.pkl")
		dfs1.append(df1)
		dfs2.append(df2)
		dfs3.append(df3)
	except:
		print(f"missing {model_type} {sid}")
performance = pd.concat(dfs1, ignore_index=True)
params = pd.concat(dfs2, ignore_index=True)
reruns = pd.concat(dfs3, ignore_index=True)
performance.to_pickle(f"data/{model_type}_{label}_performance.pkl")
params.to_pickle(f"data/{model_type}_{label}_params.pkl")
if model_type in ["NEF_WM", "NEF_RL"]:
	reruns.to_pickle(f"data/{model_type}_{label}_estimates.pkl")
else:
	reruns.to_pickle(f"data/{model_type}_{label}_reruns.pkl")
