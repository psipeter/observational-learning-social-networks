import sys
import pandas as pd
import subprocess

model_type = sys.argv[1]
label = sys.argv[2]
if len(sys.argv)>2:
	noise = True

sids = pd.read_pickle("data/human.pkl")['sid'].unique()
dfs1 = []
dfs2 = []
dfs3 = []
dfs4 = []
dfs5 = []

for sid in sids:
	try:
		df1 = pd.read_pickle(f"data/{model_type}_{sid}_performance.pkl")
		df2 = pd.read_pickle(f"data/{model_type}_{sid}_params.pkl")
		df3 = pd.read_pickle(f"data/{model_type}_{sid}_dynamics.pkl")
		if model_type in ['NEF_WM', 'NEF_RL']: 
			df4 = pd.read_pickle(f"data/{model_type}_{sid}_activities.pkl")
		if noise:
			df5 = pd.read_pickle(f"data/{model_type}_{sid}_noise.pkl")
		dfs1.append(df1)
		dfs2.append(df2)
		dfs3.append(df3)
		if model_type in ['NEF_WM', 'NEF_RL']: 
			dfs4.append(df4)
		if noise:
			dfs5.append(df5)
	except:
		print(f"missing {model_type} {sid}")

performance = pd.concat(dfs1, ignore_index=True)
params = pd.concat(dfs2, ignore_index=True)
reruns = pd.concat(dfs3, ignore_index=True)
if model_type in ['NEF_WM', 'NEF_RL']: 
	activities = pd.concat(dfs4, ignore_index=True)
if noise:
	noise_reruns = pd.concat(dfs5, ignore_index=True)

performance.to_pickle(f"data/{model_type}_{label}_performance.pkl")
params.to_pickle(f"data/{model_type}_{label}_params.pkl")
reruns.to_pickle(f"data/{model_type}_{label}_dynamics.pkl")
if model_type in ['NEF_WM', 'NEF_RL']: 
	activities.to_pickle(f"data/{model_type}_{label}_activities.pkl")
if noise:
	noise_reruns.to_pickle(f"data/{model_type}_{label}_noise.pkl")
