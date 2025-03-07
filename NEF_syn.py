import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import nengo
import scipy
import pandas as pd

class Environment():
	def __init__(self, dataset, sid, trial, T=1, dt=0.001, dim_context=5, seed_env=0):
		self.T = T
		self.dt = dt
		self.sid = sid
		self.trial = trial
		self.dataset = dataset
		self.empirical = pd.read_pickle(f"data/{dataset}.pkl").query("sid==@sid & trial==@trial")
		self.dim_context = dim_context
		self.rng = np.random.RandomState(seed=seed_env)
		self.context = self.rng.rand(self.dim_context)
		self.context = self.context / np.linalg.norm(self.context)
		if self.dataset=='carrabin':
			self.Tall = 5*self.T - self.dt
			self.stages = range(1, 6)
		# create input arrays
		self.colors = []
		self.weights = []
		tt = int(self.T / self.dt / 2)
		if self.dataset=='carrabin':
			for stage in self.stages:
				color = self.empirical.query("stage==@stage")['color'].unique()[0]
				self.colors.extend(color * np.ones((tt, 1)))
				self.colors.extend(0.000 * np.ones((tt, 1)))
		self.colors = np.array(self.colors).flatten()

	def sample(self, t):
		tidx = int(t/self.dt)
		return [self.colors[tidx]]


def simulate_NEF_syn(learned_weights, env, n_neurons=1000, seed_net=0, syn=0.01, syn_fb=0.2,
					  radius=1, a=1e-4, alpha=6e-5, dt=0.001, lambd=0, train=False, plot=False):

	nengo.rc.set("decoder_cache", "enabled", "False")
	func_stim = lambda t: env.sample(t)
	func_context = lambda t: env.context
	func_weight = lambda t: np.power(int(t)+3, -lambd) if t!=-3 else 1
	func_stop = lambda t: 1 - np.abs(env.sample(t))

	w_inh_weight = -1000*np.ones((n_neurons, 1))
	w_inh_value = -1000*np.ones((int(n_neurons/2), 1))
	
	network = nengo.Network(seed=seed_net)
	with network:
		node_stim = nengo.Node(func_stim)
		node_target = nengo.Node(func_weight)
		node_stop = nengo.Node(func_stop)
		node_context = nengo.Node(func_context)
		
		stim = nengo.Ensemble(n_neurons, 1, seed=seed_net)
		delta = nengo.Ensemble(n_neurons, 1, encoders=nengo.dists.Choice([[1]]), intercepts=nengo.dists.Uniform(0,1), seed=seed_net)
		memory = nengo.Ensemble(n_neurons, 1, radius=radius, seed=seed_net)	
		weight = nengo.Ensemble(n_neurons, 1, seed=seed_net)
		context = nengo.Ensemble(n_neurons, env.dim_context, seed=seed_net)
		value = nengo.Ensemble(n_neurons, 1, seed=seed_net)
		if train:
			error_weight = nengo.Ensemble(n_neurons, 1, seed=seed_net)
		else:
			error_value = nengo.networks.Product(n_neurons, 1, seed=seed_net)

		nengo.Connection(node_stim, stim, seed=seed_net)
		nengo.Connection(node_context, context, seed=seed_net)
		nengo.Connection(stim, delta, synapse=syn, function=lambda x: 2*np.abs(x), seed=seed_net)
		nengo.Connection(stim, delta, synapse=0.1, function=lambda x: -2*np.abs(x), seed=seed_net)
		nengo.Connection(delta, memory, synapse=syn_fb, transform=syn_fb, seed=seed_net)
		nengo.Connection(memory, memory, synapse=syn_fb, seed=seed_net)
		if train:
			conn_weight = nengo.Connection(memory.neurons, weight, synapse=syn, seed=seed_net,
										   transform=learned_weights, learning_rule_type=nengo.PES(learning_rate=a))
			nengo.Connection(node_target, error_weight, transform=1, seed=seed_net)
			nengo.Connection(weight, error_weight, transform=-1, seed=seed_net)
			nengo.Connection(error_weight, conn_weight.learning_rule, transform=-1, seed=seed_net)
			nengo.Connection(node_stop, error_weight.neurons, transform=w_inh_weight, seed=seed_net)
		else:
			conn_weight = nengo.Connection(memory.neurons, weight, transform=learned_weights, synapse=syn, seed=seed_net)
			conn_value = nengo.Connection(context, value, synapse=syn, seed=seed_net,
										  learning_rule_type=nengo.PES(learning_rate=alpha), function=lambda x: 0)
			nengo.Connection(stim, error_value.input_a, synapse=syn, seed=seed_net)
			nengo.Connection(value, error_value.input_a, transform=-1, synapse=syn, seed=seed_net)
			nengo.Connection(weight, error_value.input_b, synapse=syn, seed=seed_net)
			nengo.Connection(error_value.output, conn_value.learning_rule, synapse=syn, transform=-1, seed=seed_net)
			nengo.Connection(node_stop, error_value.sq1.ea_ensembles[0].neurons, transform=w_inh_value, seed=seed_net)
			nengo.Connection(node_stop, error_value.sq2.ea_ensembles[0].neurons, transform=w_inh_value, seed=seed_net)

		probe_stim = nengo.Probe(stim, synapse=syn)
		probe_target = nengo.Probe(node_target, synapse=syn)
		probe_delta = nengo.Probe(delta, synapse=syn)
		probe_memory = nengo.Probe(memory, synapse=syn)
		probe_weight = nengo.Probe(weight, synapse=syn)
		probe_value = nengo.Probe(value, synapse=syn)
		if train:
			probe_error_weight = nengo.Probe(error_weight, synapse=0.01)
			probe_learned_weights = nengo.Probe(conn_weight, "weights")
		else:
			probe_error_value = nengo.Probe(error_value.output, synapse=0.01)
		network.probe_target = probe_target
		network.probe_weight = probe_weight
		network.probe_value = probe_value

	sim = nengo.Simulator(network, dt=dt, progress_bar=False)
	with sim:
		sim.run(5-dt)

	if train:
		learned_weights = sim.data[probe_learned_weights][-1]
		return learned_weights
	else:
		return network, sim


def run_NEF_syn(dataset, sid, alpha, z, lambd, n_neurons=200):
	empirical = pd.read_pickle(f"data/{dataset}.pkl").query("sid==@sid")
	trials = empirical['trial'].unique() 
	columns = ['type', 'sid', 'trial', 'stage', 'estimate']
	dfs = []
	# train mapping from count memory to weight
	W = np.zeros((1, n_neurons))
	print(f"training sid {sid}")
	for trial in trials[:50]:
		env = Environment('carrabin', sid=sid, trial=trial)
		W = simulate_NEF_syn(W, env, alpha=alpha, n_neurons=n_neurons, lambd=lambd, seed_net=sid, train=True)
	# run the trials with the learned weights
	for trial in trials:
		print(f"sid {sid}, trial {trial}")
		env = Environment('carrabin', sid=sid, trial=trial)
		net, sim = simulate_NEF_syn(W, env, alpha=alpha, n_neurons=n_neurons, lambd=lambd, seed_net=sid, train=False)
		for stage in env.stages:
			if dataset=='carrabin':
				tidx = int((stage*env.T)/env.dt)-2
				estimate = np.mean(sim.data[net.probe_value][tidx-100: tidx])
				df = pd.DataFrame([['NEF_syn', sid, trial, stage, estimate]], columns=columns)
				dfs.append(df)
	data = pd.concat(dfs, ignore_index=True)
	data.to_pickle(f"data/NEF_syn_{dataset}_{sid}_estimates.pkl")
	return data