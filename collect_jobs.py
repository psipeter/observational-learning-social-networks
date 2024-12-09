import sys
import pandas as pd
import subprocess

model_type = sys.argv[1]
label = sys.argv[2]
sids = pd.read_pickle("data/human.pkl")['sid'].unique()
dfs1 = []
dfs2 = []
dfs3 = []

if model_type in ['all', 'NEF-WM', 'NEF-RL', 'RL1', 'RL3', 'RL3rd', 'ZK', 'DGn', 'DGrd', 'DGrds', 'DGrdp', 'DGrdpz']:
	if model_type == 'all':
		for mt in ['RL1', 'RL3rd', 'DGn', 'ZK', 'NEF-WM', 'NEF-RL']:
			for sid in sids:
				try:
					df1 = pd.read_pickle(f"data/{mt}_{sid}_performance.pkl")
					df2 = pd.read_pickle(f"data/{mt}_{sid}_params.pkl")
					df3 = pd.read_pickle(f"data/{mt}_{sid}_rerun.pkl")
					dfs1.append(df1)
					dfs2.append(df2)
					dfs3.append(df3)
				except:
					print(f"missing {mt} {sid}")
		dfs3.append(pd.read_pickle("data/human.pkl"))
	else:
		for sid in sids:
			try:
				df1 = pd.read_pickle(f"data/{model_type}_{sid}_performance.pkl")
				df2 = pd.read_pickle(f"data/{model_type}_{sid}_params.pkl")
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
	reruns.to_pickle(f"data/{model_type}_{label}_reruns.pkl")

if model_type in ['WM', 'RL']:
	for sid in sids:
		try:
			df1 = pd.read_pickle(f"data/{model_type}_{sid}.pkl")
			dfs1.append(df1)
		except:
			print(f"missing {model_type} {sid}")
	estimates = pd.concat(dfs1, ignore_index=True)
	estimates.to_pickle(f"data/{model_type}_{label}_estimates.pkl")
