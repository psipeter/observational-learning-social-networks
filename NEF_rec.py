import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import nengo
import scipy
import pandas as pd
from uniform_encoders import *
from environments import *

def simulate_NEF_rec(learned_weights, env, n_neurons=1000, seed_net=0, syn=0.01, syn_fb=0.2, x_int=0.5, syn_spike=0.01, 
					  a=1e-4, alpha=0.2, dt=0.001, z=0, train=False, plot=False):
	nengo.rc.set("decoder_cache", "enabled", "False")
	func_stim = lambda t: env.sample_color(t)
	func_weight = lambda t: env.sample_weight(t)
	func_noise = lambda t: env.sample_noise(t)
	func_stop = lambda t: 1 if env.sample_color(t)==0 else 0
	func_neighbor_degree = lambda t: env.sample_neighbor_degree(t)

	w_noise = np.ones((n_neurons, 1))
	w_inh_weight = -1000*np.ones((n_neurons, 1))
	w_inh_update = -1000*np.ones((int(n_neurons/2), 1))
	if env.dataset=='carrabin':
		radius = 1
	if env.dataset=='jiang':
		radius = 2
	if env.dataset=='yoo':
		radius = 5
	func_project = lambda x: [np.sin(np.pi*x/radius), np.cos(np.pi*x/radius)]

	network = nengo.Network(seed=seed_net)
	with network:
		node_stim = nengo.Node(func_stim)
		node_target = nengo.Node(func_weight)
		node_neighbor_degree = nengo.Node(func_neighbor_degree)
		node_stop = nengo.Node(func_stop)
		node_noise = nengo.Node(func_noise)
		
		stim = nengo.Ensemble(n_neurons, 1, seed=seed_net)
		delta = nengo.Ensemble(n_neurons, 1, encoders=nengo.dists.Choice([[1]]), intercepts=nengo.dists.Uniform(0,1), seed=seed_net)
		memory = nengo.Ensemble(n_neurons, 1, radius=radius, seed=seed_net)	
		project = nengo.Ensemble(n_neurons, 2, intercepts=nengo.dists.Uniform(x_int, 1), seed=seed_net)
		weight = nengo.Ensemble(n_neurons, 1, seed=seed_net) # radius=radius, 
		neighbor_degree = nengo.Ensemble(n_neurons, 1, seed=seed_net)
		value = nengo.Ensemble(n_neurons, 1, seed=seed_net)
		if train:
			error_weight = nengo.Ensemble(n_neurons, 1, seed=seed_net)
		else:
			error_value = nengo.networks.Product(n_neurons, 1, seed=seed_net)

		nengo.Connection(node_stim, stim, seed=seed_net)
		nengo.Connection(node_neighbor_degree, neighbor_degree, seed=seed_net)
		nengo.Connection(node_noise, value.neurons, transform=w_noise, seed=seed_net)
		nengo.Connection(stim, delta, synapse=syn, function=lambda x: 2*np.abs(x), seed=seed_net)
		nengo.Connection(stim, delta, synapse=0.1, function=lambda x: -2*np.abs(x), seed=seed_net)
		# nengo.Connection(delta, memory, synapse=syn_fb, transform=syn_fb, seed=seed_net)
		nengo.Connection(delta, memory, synapse=syn_fb, function=lambda x: 0.1, seed=seed_net)
		nengo.Connection(memory, memory, synapse=syn_fb, seed=seed_net)
		nengo.Connection(memory, project, synapse=syn, function=func_project, seed=seed_net)
		nengo.Connection(value, value, synapse=syn_fb, seed=seed_net)
		if train:
			conn_weight = nengo.Connection(project.neurons, weight,
											transform=learned_weights, learning_rule_type=nengo.PES(learning_rate=a), seed=seed_net)
			nengo.Connection(node_target, error_weight, transform=1, seed=seed_net)
			nengo.Connection(weight, error_weight, transform=-1, seed=seed_net)
			nengo.Connection(error_weight, conn_weight.learning_rule, transform=-1, seed=seed_net)
			nengo.Connection(node_stop, error_weight.neurons, transform=w_inh_weight, seed=seed_net)
		else:
			conn_weight = nengo.Connection(project.neurons, weight, transform=learned_weights, seed=seed_net)                     
			nengo.Connection(neighbor_degree, weight, transform=z, seed=seed_net)                     
			nengo.Connection(stim, error_value.input_a, synapse=syn)
			nengo.Connection(value, error_value.input_a, transform=-1, synapse=syn, seed=seed_net)
			nengo.Connection(weight, error_value.input_b, synapse=syn, seed=seed_net)
			nengo.Connection(error_value.output, value, transform=alpha, synapse=syn_fb, seed=seed_net)
			nengo.Connection(node_stop, error_value.sq1.ea_ensembles[0].neurons, transform=w_inh_update, seed=seed_net)
			nengo.Connection(node_stop, error_value.sq2.ea_ensembles[0].neurons, transform=w_inh_update, seed=seed_net)

		probe_node_stim = nengo.Probe(node_stim, synapse=None)
		probe_target = nengo.Probe(node_target, synapse=None)
		probe_neighbor_degree = nengo.Probe(node_neighbor_degree, synapse=None)
		probe_stim = nengo.Probe(stim, synapse=syn)
		probe_delta = nengo.Probe(delta, synapse=syn)
		probe_memory = nengo.Probe(memory, synapse=syn)
		probe_weight = nengo.Probe(weight, synapse=syn)
		probe_project = nengo.Probe(project, synapse=syn)
		probe_value = nengo.Probe(value, synapse=syn)
		if train:
			network.probe_error_weight = nengo.Probe(error_weight, synapse=syn)
			probe_learned_weights = nengo.Probe(conn_weight, "weights")
		else:
			network.probe_error_value = nengo.Probe(error_value.output, synapse=syn)
			network.probe_error1_spikes = nengo.Probe(error_value.sq1.ea_ensembles[0].neurons, synapse=syn)
			network.probe_error2_spikes = nengo.Probe(error_value.sq2.ea_ensembles[0].neurons, synapse=syn)
		network.probe_memory = probe_memory
		network.probe_node_stim = probe_node_stim
		network.probe_stim = probe_stim
		network.probe_neighbor_degree = probe_neighbor_degree
		network.probe_target = probe_target
		network.probe_weight = probe_weight
		network.probe_project = probe_project
		network.probe_value = probe_value
		network.probe_stim_spikes = nengo.Probe(stim.neurons, synapse=syn_spike)
		network.probe_weight_spikes = nengo.Probe(weight.neurons, synapse=syn_spike)
		network.probe_value_spikes = nengo.Probe(value.neurons, synapse=syn_spike)

	sim = nengo.Simulator(network, dt=dt, progress_bar=False)
	with sim:
		sim.run(env.Tall)

	if train:
		learned_weights = sim.data[probe_learned_weights][-1]
		return network, sim, learned_weights
	else:
		return network, sim


def run_NEF_rec(dataset, sid, alpha, z, lambd, n_neurons=200, pretrain=True, iti_decode=False, iti_noise=0):
	empirical = pd.read_pickle(f"data/{dataset}.pkl").query("sid==@sid")
	trials = empirical['trial'].unique() 
	columns = ['type', 'sid', 'trial', 'stage', 'estimate']
	dfs = []
	if pretrain:
		W = np.zeros((1, n_neurons))
		for trial in trials[:20]:
			print(f"training sid {sid}, trial {trial}")
			env = EnvironmentCount(dataset, sid=sid, trial=trial, lambd=lambd, iti_noise=iti_noise)
			net, sim, W = simulate_NEF_rec(W, env, alpha=alpha, n_neurons=n_neurons, z=z, seed_net=sid, train=True)
		np.savez(f"data/NEF_rec_{dataset}_{sid}_pretrained_weight.npz", W=W)
	else:
		W = np.load(f"data/NEF_rec_{dataset}_{sid}_pretrained_weight.npz")['W']
	for trial in trials:
		print(f"running sid {sid}, trial {trial}")
		env = EnvironmentCount(dataset, sid=sid, trial=trial, lambd=lambd, iti_noise=iti_noise)
		net, sim = simulate_NEF_rec(W, env, alpha=alpha, n_neurons=n_neurons, z=z, seed_net=sid, train=False)
		obs_times = env.obs_times if not iti_decode else env.iti_times
		for s, tidx in enumerate(obs_times):
			stage = env.stages[s]
			estimate = np.mean(sim.data[net.probe_value][tidx-100: tidx])
			dfs.append(pd.DataFrame([['NEF_rec', sid, trial, stage, estimate]], columns=columns))
	data = pd.concat(dfs, ignore_index=True)
	data.to_pickle(f"data/NEF_rec_{dataset}_{sid}_estimates.pkl")
	return data

def activities_NEF_rec(dataset, sid, alpha, z, lambd, n_neurons=500, pretrain=True):
	empirical = pd.read_pickle(f"data/{dataset}.pkl").query("sid==@sid")
	trials = empirical['trial'].unique() 
	columns = ['type', 'trial', 'stage', 'population', 'neuron', 'aPE', 'RD', 'lambd', 'activity']
	dfs = []
	if pretrain:
		W = np.zeros((1, n_neurons))
		for trial in trials[:20]:
			print(f"training sid {sid}, trial {trial}")
			env = EnvironmentCount(dataset, sid=sid, trial=trial, lambd=lambd)
			net, sim, W = simulate_NEF_rec(W, env, alpha=alpha, n_neurons=n_neurons, z=z, seed_net=sid, train=True)
		np.savez(f"data/NEF_rec_{dataset}_{sid}_pretrained_weight.npz", W=W)
	else:
		W = np.load(f"data/NEF_rec_{dataset}_{sid}_pretrained_weight.npz")['W']
	for trial in trials:
		print(f"running sid {sid}, trial {trial}")
		env = EnvironmentCount(dataset, sid=sid, trial=trial, lambd=lambd)
		net, sim = simulate_NEF_rec(W, env, alpha=alpha, n_neurons=n_neurons, z=z, seed_net=sid, train=False)
		obs_times = np.arange(3, 3+3*env.n_neighbors+1, 1) * env.T/env.dt - env.T/env.dt/2
		obs_times = obs_times.astype(int)
		for s, tidx in enumerate(obs_times):
			obs = sim.data[net.probe_node_stim][tidx-10]
			estimate = sim.data[net.probe_value][tidx-10]
			aPE = np.abs(obs - estimate)[0]
			# RD = np.mean(sim.data[net.probe_neighbor_degree][tidx-100: tidx])
			RD = empirical.query("trial==@trial")['RD'].to_numpy()[s]
			stage = empirical.query("trial==@trial")['stage'].to_numpy()[s]
			for pop in ['weight', 'error1', 'error2']:
				if pop=='weight': activity = sim.data[net.probe_weight_spikes][tidx-10]
				if pop=='error1': activity = sim.data[net.probe_error1_spikes][tidx-10]
				if pop=='error2': activity = sim.data[net.probe_error2_spikes][tidx-10]
				neurons = np.arange(1, activity.shape[0]+1, 1)
				df = pd.DataFrame(columns=columns)
				df['neuron'] = neurons
				df['activity'] = np.around(activity, 4)
				df['population'] = pop
				df['type'] = "NEF_rec"
				df['trial'] = trial
				df['stage'] = stage
				df['aPE'] = aPE
				df['RD'] = RD
				df['lambd'] = lambd
				dfs.append(df)
	data = pd.concat(dfs, ignore_index=True)
	data.to_pickle(f"data/NEF_rec_{dataset}_{sid}_activities.pkl")
	return data