import numpy as np
import scipy
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import sys
from fit import get_expectations_carrabin, get_expectations_jiang
import time

def rerun_carrabin(model_type, sid):
	human = pd.read_pickle(f"data/carrabin.pkl").query("sid==@sid")
	trials = human['trial'].unique()
	stages = human['stage'].unique()
	params = pd.read_pickle(f"data/{model_type}_carrabin_{sid}_params.pkl").loc[0].to_numpy()[2:]
	dfs = []
	columns = ['type', 'sid', 'trial', 'stage', 'qid', 'response']
	for trial in trials:
		for stage in stages:
			qid = human.query("trial==@trial and stage==@stage")['qid'].unique()[0]
			response = get_expectations_carrabin(model_type, params, sid, trial, stage)
			dfs.append(pd.DataFrame([[model_type, sid, trial, stage, qid, response]], columns=columns))
	dynamics_data = pd.concat(dfs, ignore_index=True)
	dynamics_data.to_pickle(f"data/{model_type}_carrabin_{sid}_dynamics.pkl")
	return dynamics_data

def rerun_jiang(model_type, sid, seed=0):
	human = pd.read_pickle(f"data/jiang.pkl").query("sid==@sid")
	trials = human['trial'].unique()
	stages = human['stage'].unique()
	rng = np.random.RandomState(seed=seed)
	params = pd.read_pickle(f"data/{model_type}_jiang_{sid}_params.pkl").loc[0].to_numpy()[2:]
	beta = params[-1]
	dfs = []
	columns = ['type', 'sid', 'trial', 'network', 'stage', 'who', 'color', 'degree', 'RD', 'action']
	for trial in trials:
		for stage in stages:
			expectation = get_expectations_jiang(model_type, params, sid, trial, stage)
			prob = scipy.special.expit(beta*expectation)
			action = 1 if rng.uniform(0,1) < prob else -1
			subdata = human.query("trial==@trial & stage==@stage")
			observations = subdata['color'].to_numpy()
			for o, obs in enumerate(observations):
				network = subdata['network'].unique()[0]
				who = subdata['who'].to_numpy()[o]
				color = subdata['color'].to_numpy()[o]
				degree = subdata['degree'].to_numpy()[o]
				RD = subdata['RD'].to_numpy()[o]
				dfs.append(pd.DataFrame([[model_type, sid, trial, network, stage, who, color, degree, RD, action]], columns=columns))
	dynamics_data = pd.concat(dfs, ignore_index=True)
	dynamics_data.to_pickle(f"data/{model_type}_jiang_{sid}_dynamics.pkl")
	return dynamics_data

def rerun_yo(model_type, sid):
	human = pd.read_pickle(f"data/yoo.pkl").query("sid==@sid")
	params = pd.read_pickle(f"data/{model_type}_yoo_{sid}_params.pkl").loc[0].to_numpy()[2:]
	dfs = []
	columns = ['type', 'sid', 'block', 'trial', 'stage', 'response']
	for sid in df['sid'].unique():
		for block in human.query("sid==@sid")['block'].unique():
			for trial in human.query("sid==@sid & block==@block")['trial'].unique():
				for stage in human.query("sid==@sid & block==@block & trial==@trial")['stage'].unique():
					subdata = human.query("sid==@sid & block==@block & trial==@trial")
					response = get_expectations_yoo(model_type, params, sid, block, trial, stage, subdata)
					dfs.append(pd.DataFrame([[model_type, sid, block, trial, stage, response]], columns=columns))
	dynamics_data = pd.concat(dfs, ignore_index=True)
	dynamics_data.to_pickle(f"data/{model_type}_yoo_{sid}_dynamics.pkl")
	return dynamics_data

if __name__ == '__main__':
	dataset = sys.argv[1]
	model_type = sys.argv[2]
	sid = int(sys.argv[3])
	start = time.time()
	if dataset=='carrabin':
		choice_data = rerun_carrabin(model_type, sid)
	elif dataset=='jiang':
		choice_data = rerun_jiang(model_type, sid)
	print(choice_data)
	end = time.time()
	print(f"runtime {(end-start)/60:.4} min")