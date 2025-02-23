import numpy as np
import scipy
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import sys
from fit import get_expectations_carrabin, get_expectations_jiang, likelihood, compute_mcfadden
import time

def rerun_carrabin(model_type, sid):
	human = pd.read_pickle(f"data/carrabin.pkl").query("sid==@sid")
	trials = human['trial'].unique()
	stages = human['stage'].unique()
	if model_type in ['bayes', 'bayesPE']:
		params = []
	else:
		params = pd.read_pickle(f"data/{model_type}_{dataset}_{sid}_params.pkl").loc[0].to_numpy()[2:]
	dfs = []
	columns = ['type', 'sid', 'trial', 'stage', 'qid', 'response']
	for trial in trials:
		for stage in stages:
			qid = human.query("trial==@trial and stage==@stage")['qid'].unique()[0]
			response = get_expectations_carrabin(model_type, params, sid, trial, stage)
			dfs.append(pd.DataFrame([[model_type, sid, trial, stage, qid, response]], columns=columns))
	dynamics_data = pd.concat(dfs, ignore_index=True)
	dynamics_data.to_pickle(f"data/{model_type}_{dataset}_{sid}_dynamics.pkl")
	return dynamics_data

def rerun_jiang(model_type, sid, seed=0, noise=False, sigma=0):
	rng = np.random.RandomState(seed=seed)
	human = pd.read_pickle(f"data/jiang.pkl").query("sid==@sid")
	trials = human['trial'].unique()
	stages = human['stage'].unique()
	params = pd.read_pickle(f"data/{model_type}_{sid}_params.pkl").loc[0].to_numpy()[2:]
	inv_temp = params[-1]
	dfs = []
	columns = ['type', 'sid', 'trial', 'network', 'stage', 'who', 'color', 'degree', 'RD', 'action', 'expectation']
	for trial in trials:
		for stage in stages:
			expectations = get_expectations_jiang(model_type, params, trial, stage, sid, noise=noise, sigma=sigma, rng=rng)
			final_expectation = expectations[-1]
			prob = scipy.special.expit(inv_temp*final_expectation)
			action = 1 if rng.uniform(0,1) < prob else -1
			subdata = human.query("trial==@trial & stage==@stage")
			observations = subdata['color'].to_numpy()
			# NOTE: action and expectation will be identical for all observations within a stage
			# (that is, they are the 'final' expectation and action for this stage)
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
					action,
					final_expectation
				]], columns=columns)
				dfs.append(df)
	dynamics_data = pd.concat(dfs, ignore_index=True)
	dynamics_data.to_pickle(f"data/{model_type}_{dataset}_{sid}_dynamics.pkl")
	return dynamics_data


def get_mean_accuracy(data, sid):
	human2 = pd.read_pickle("data/jiang2.pkl").query("sid==@sid")
	corrects = []
	for trial in human2['trial'].unique():
		sum_private = human2.query("trial==@trial")['sum private'].to_numpy()[-1]
		action = data.query("sid==@sid & trial==@trial")['action'].to_numpy()[-1]
		correct = 1 if np.sign(action)==np.sign(sum_private) else 0
		corrects.append(correct)
	return np.mean(corrects)


def noise_rerun(model_type, sid, sigmas, seed=0):
	rng = np.random.RandomState(seed=seed)
	params = pd.read_pickle(f"data/{model_type}_{sid}_params.pkl").loc[0].to_numpy()[2:]
	dfs = []
	columns = ['type', 'sid', 'sigma', 'NLL', 'McFadden R2', 'mean accuracy']
	for sigma in sigmas:
		dynamics = rerun_jiang(model_type, sid, seed=seed, noise=True, sigma=sigma)
		NLL = likelihood(params, model_type, sid, noise=True, sigma=sigma)
		mcfadden_r2 = compute_mcfadden(NLL, sid)
		mean_accuracy = get_mean_accuracy(dynamics, sid)
		dfs.append(pd.DataFrame([[model_type, sid, sigma, NLL, mcfadden_r2, mean_accuracy]], columns=columns))
	noise_data = pd.concat(dfs, ignore_index=True)
	noise_data.to_pickle(f"data/{model_type}_{dataset}_{sid}_noise.pkl")
	return noise_data

if __name__ == '__main__':
	dataset = sys.argv[1]
	model_type = sys.argv[2]
	sid = int(sys.argv[3])
	# sigmas = np.arange(0, 1.025, 0.025)
	start = time.time()
	if dataset=='carrabin':
		choice_data = rerun_carrabin(model_type, sid)
		print(choice_data)
	elif dataset=='jiang':
		choice_data = rerun_jiang(model_type, sid)
		# noise_data = noise_rerun(model_type, sid, sigmas)
		print(choice_data)
		# print(noise_data)
	end = time.time()
	print(f"runtime {(end-start)/60:.4} min")