import numpy as np
import scipy
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import sys
from fit import get_expectations, likelihood, compute_mcfadden
import time

def rerun(model_type, sid, seed=0):
	rng = np.random.RandomState(seed=seed)
	human = pd.read_pickle(f"data/human.pkl").query("sid==@sid")
	trials = human['trial'].unique()
	stages = human['stage'].unique()
	params = pd.read_pickle(f"data/{model_type}_{sid}_params.pkl").loc[0].to_numpy()[2:]
	inv_temp = params[-1]
	dfs = []
	columns = ['type', 'sid', 'trial', 'network', 'stage', 'who', 'color', 'degree', 'RD', 'action', 'expectation']
	for trial in trials:
		for stage in stages:
			expectations = get_expectations(model_type, params, trial, stage, sid)
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
	dynamics_data.to_pickle(f"data/{model_type}_{sid}_dynamics.pkl")
	return dynamics_data


def noise_rerun(model_type, sid, sigmas, seed=0):
	rng = np.random.RandomState(seed=seed)
	dfs = []
	columns = ['type', 'sid', 'sigma', 'NLL', 'McFadden R2']
	for sigma in sigmas:
		NLL = likelihood(params, model_type, sid, noise=True, sigma=sigma)
		mcfadden_r2 = compute_mcfadden(NLL, sid)
		dfs.append(pd.DataFrame([[model_type, sid, sigma, NLL, mcfadden_r2]], columns=columns))
	noise_data = pd.concat(dfs, ignore_index=True)
	noise_data.to_pickle(f"data/{model_type}_{sid}_noise.pkl")
	return noise_data

if __name__ == '__main__':
	model_type = sys.argv[1]
	sid = int(sys.argv[2])
	sigmas = np.arange(0, 0.5, 0.01)
	start = time.time()
	choice_data = rerun(model_type, sid)
	noise_data = noise_rerun(model_type, sid, sigmas)
	print(choice_data)
	end = time.time()
	print(f"runtime {(end-start)/60:.4} min")