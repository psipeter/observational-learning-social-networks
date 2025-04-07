import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import nengo
import scipy
import time
import sys
from fit import *
from NEF_WM import *
from NEF_RL import *
from NEF_syn import *
from environments import *

if __name__ == '__main__':
	model_type = sys.argv[1]
	sid = int(sys.argv[2])
	if model_type=='NEF_syn':
		p_nef = pd.read_pickle("data/NEF_syn_jiang_apr7lambda0_params.pkl").query("sid==@sid")
		n_neurons = p_nef['n_neurons'].unique()[0]
		alpha = p_nef['alpha'].unique()[0]
		z = p_nef['z'].unique()[0]
		lambd = p_nef['lambda'].unique()[0]
		start = time.time()
		activity_data = activities_NEF_syn(dataset='jiang', sid=sid, alpha=alpha, z=z, lambd=lambd, n_neurons=n_neurons)
		print(activity_data)
		# activity_data.to_pickle(f"data/{model_type}_{sid}_activities.pkl")
		end = time.time()
		print(f"runtime {(end-start)/60:.4} min")
	else:
		empirical = pd.read_pickle(f"data/jiang.pkl").query("sid==@sid")
		trials = empirical['trial'].unique()
		rng = np.random.RandomState(seed=0)
		dfs = []
		columns = ['type', 'trial', 'stage', 'population', 'neuron', 'aPE', 'RD', 'activity']
		start = time.time()
		if model_type=='NEF_RL':
			p_nef = pd.read_pickle("data/NEF_RL_jiang_mar11_params.pkl").query("sid==@sid")
			alpha = p_nef['alpha'].unique()[0]
			z = p_nef['z'].unique()[0]
			lambd = p_nef['lambda'].unique()[0]
		if model_type=='NEF_WM':
			p_nef = pd.read_pickle("data/NEF_WM_jiang_mar11_params.pkl").query("sid==@sid")
			alpha = p_nef['alpha'].unique()[0]
			z = p_nef['z'].unique()[0]
			lambd = p_nef['lambda'].unique()[0]		
		for trial in trials:
			print(f"trial {trial}")
			seed_net = sid + 1000*trial
			if model_type=='NEF_RL':
				env = EnvironmentRL(dataset="jiang", sid=sid, trial=trial, alpha=alpha, z=z, lambd=lambd)
				net = build_network_RL(env, seed_net=seed_net)
			if model_type=='NEF_WM':
				env = EnvironmentWM(dataset="jiang", sid=sid, trial=trial, alpha=alpha, z=z, lambd=lambd)
				net = build_network_WM(env, seed_net=seed_net)
			sim = nengo.Simulator(net, seed=sid, progress_bar=False)
			with sim:
				sim.run(env.Tall, progress_bar=False)
			obs_times = np.arange(3, 3+3*env.n_neighbors+1, 1) * env.T/env.dt
			obs_times = obs_times.astype(int)
			for s, tidx in enumerate(obs_times):
				obs = np.mean(sim.data[net.probe_input_obs][tidx-100: tidx])
				estimate = np.mean(sim.data[net.probe_value][tidx-100: tidx])
				aPE = np.abs(obs - estimate)
				RD = empirical.query("trial==@trial")['RD'].to_numpy()[s]
				stage = empirical.query("trial==@trial")['stage'].to_numpy()[s]
				for pop in ['weight', 'error1', 'error2']:
					if pop=='weight': activity = np.mean(sim.data[net.probe_weight_spikes][tidx-100: tidx], axis=0)
					if pop=='error1': activity = np.mean(sim.data[net.probe_error1_spikes][tidx-100: tidx], axis=0)
					if pop=='error2': activity = np.mean(sim.data[net.probe_error2_spikes][tidx-100: tidx], axis=0)
					neurons = np.arange(1, activity.shape[0]+1, 1)
					df = pd.DataFrame(columns=columns)
					df['neuron'] = neurons
					df['activity'] = np.around(activity, 4)
					df['population'] = pop
					df['type'] = model_type
					df['trial'] = trial
					df['stage'] = stage
					df['aPE'] = aPE
					df['RD'] = RD
					dfs.append(df)
		activity_data = pd.concat(dfs, ignore_index=True)
		activity_data.to_pickle(f"data/{model_type}_{sid}_activities.pkl")
		end = time.time()
		print(f"runtime {(end-start)/60:.4} min")