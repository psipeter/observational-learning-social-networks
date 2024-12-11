import numpy as np
import scipy
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import sys
from fit import get_expectation
import time

def rerun(model_type, sid, save=True, seed=0):
	rng = np.random.RandomState(seed=seed)
	human = pd.read_pickle(f"data/human.pkl").query("sid==@sid")
	trials = human['trial'].unique()
	stages = human['stage'].unique()
	params = pd.read_pickle(f"data/{model_type}_{sid}_params.pkl").loc[0].to_numpy()[1:]
	dfs = []
	columns = ['type', 'sid', 'trial', 'network', 'stage', 'who', 'color', 'degree', 'RD', 'action']
	for trial in trials:
		for stage in stages:
			# update expectation based on model type
			expectation = get_expectation(model_type, params, trial, stage, sid)
			# generate one row per observation, even though 'action' will be the same
			inv_temp = params[-1]
			prob = scipy.special.expit(inv_temp*expectation)
			action = 1 if rng.uniform(0,1) < prob else -1
			subdata = human.query("trial==@trial & stage==@stage")
			observations = subdata['color'].to_numpy()
			for o, obs in enumerate(observations):
				network = subdata['network'].unique()[0]
				who = subdata['who'].to_numpy()[o]
				color = subdata['color'].to_numpy()[o]
				degree = subdata['degree'].to_numpy()[o]
				RD = subdata['RD'].to_numpy()[o]
				df = pd.DataFrame([[
					model_type,
					sid,
					trial,
					network,
					stage,
					who,
					color,
					degree,
					RD,
					action
				]], columns=columns)
				dfs.append(df)
	rerun_data = pd.concat(dfs, ignore_index=True)
	if save:
		rerun_data.to_pickle(f"data/{model_type}_{sid}_rerun.pkl")
	return rerun_data


if __name__ == '__main__':

	model_type = sys.argv[1]
	sid = int(sys.argv[2])

	start = time.time()
	if model_type=='all':
		model_types = ['RL1', 'RL3rd', 'DGn', 'ZK', 'NEF-WM', 'NEF-RL']
		for mt in model_types:
			print(f"rerunning {mt}, sid {sid}")
			choice_data = rerun(mt, sid)
	else:
		choice_data = rerun(model_type, sid)

	print(choice_data)
	end = time.time()
	print(f"runtime {(end-start)/60:.4} min")