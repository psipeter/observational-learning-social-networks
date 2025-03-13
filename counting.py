import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import nengo
import scipy
import pandas as pd
import sys
import time

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
		if self.dataset=='jiang':
			self.n_neighbors = len(self.empirical['who'].unique()) - 1
			self.Tall = 3*self.T + 3*self.n_neighbors*self.T - self.dt
			self.stages = range(4)
		# create input arrays
		self.colors = []
		self.weights = []
		tt = int(self.T / self.dt / 2)
		if self.dataset=='carrabin':
			for stage in self.stages:
				color = self.empirical.query("stage==@stage")['color'].unique()[0]
				self.colors.extend(color * np.ones((tt, 1)))
				self.colors.extend(0.000 * np.ones((tt, 1)))
		if self.dataset=='jiang':
			for stage in self.stages:
				if stage==0:
					color = self.empirical.query("stage==@stage")['color'].to_numpy()[0]
					self.colors.extend(color * np.ones((5*tt, 1)))
					self.colors.extend(0.000 * np.ones((1*tt, 1)))
				else:
					for n in range(self.n_neighbors):
						color = self.empirical.query("stage==@stage")['color'].to_numpy()[n]
						self.colors.extend(color * np.ones((tt, 1)))
						self.colors.extend(0.000 * np.ones((tt, 1)))
		self.colors = np.array(self.colors).flatten()
	def sample(self, t):
		tidx = int(t/self.dt)
		return [self.colors[tidx]]

def run(learned_weights, env, n_neurons=1000, seed_net=0, syn_fb=0.2, a=1e-4, train=False, plot=True):
	func_stim = lambda t: env.sample(t)
	func_stop = lambda t: 1 - np.abs(env.sample(t))
	func_project = lambda x: [np.sin(np.pi*x), np.cos(np.pi*x)]
	if env.dataset=='jiang':
		func_weight = lambda t: 1/(int(t)-1) if t>3 else 1
		radius = 4
		x_int = 0.5
	if env.dataset=='carrabin':
		func_weight = lambda t: 1/(int(t)+3) if t>0 else 1
		radius = 4
		x_int = 0.5

	network = nengo.Network(seed=seed_net)
	with network:
		node_stim = nengo.Node(func_stim)
		node_target = nengo.Node(func_weight)
		node_stop = nengo.Node(func_stop)
		
		ens_stim = nengo.Ensemble(n_neurons, 1, seed=seed_net)
		ens_delta = nengo.Ensemble(n_neurons, 1, encoders=nengo.dists.Choice([[1]]), intercepts=nengo.dists.Uniform(0,1), seed=seed_net)
		ens_memory = nengo.Ensemble(n_neurons, 1, radius=radius, seed=seed_net)
		ens_project = nengo.Ensemble(n_neurons, 2, intercepts=nengo.dists.Uniform(x_int, 1), seed=seed_net)
		ens_weight = nengo.Ensemble(n_neurons, 1, seed=seed_net)
		ens_error = nengo.Ensemble(n_neurons, 1, seed=seed_net)

		conn_stim = nengo.Connection(node_stim, ens_stim, seed=seed_net)
		conn_delta1 = nengo.Connection(ens_stim, ens_delta, synapse=0.01, function=lambda x: 2*np.abs(x), seed=seed_net)
		conn_delta2 = nengo.Connection(ens_stim, ens_delta, synapse=0.1, function=lambda x: -2*np.abs(x), seed=seed_net)
		conn_memff = nengo.Connection(ens_delta, ens_memory, synapse=syn_fb, transform=syn_fb, seed=seed_net)
		conn_memfb = nengo.Connection(ens_memory, ens_memory, synapse=syn_fb, seed=seed_net)
		conn_proj = nengo.Connection(ens_memory, ens_project, synapse=0.01, function=func_project, seed=seed_net)
		if train:
			conn_weight = nengo.Connection(ens_project.neurons, ens_weight,
										   transform=learned_weights, learning_rule_type=nengo.PES(learning_rate=a), seed=seed_net)
			conn_error1 = nengo.Connection(node_target, ens_error, transform=1, seed=seed_net)
			conn_error2 = nengo.Connection(ens_weight, ens_error, transform=-1, seed=seed_net)
			conn_learning = nengo.Connection(ens_error, conn_weight.learning_rule, transform=-1, seed=seed_net)
			conn_stop = nengo.Connection(node_stop, ens_error.neurons, transform=-1000*np.ones((n_neurons, 1)), seed=seed_net)
		else:
			conn_weight = nengo.Connection(ens_project.neurons, ens_weight, transform=learned_weights, seed=seed_net)					 

		probe_node_stim = nengo.Probe(node_stim, synapse=None)
		probe_stim = nengo.Probe(ens_stim, synapse=0.01)
		probe_target = nengo.Probe(node_target, synapse=None)
		# probe_target = nengo.Probe(node_target, synapse=syn_fb)
		probe_delta = nengo.Probe(ens_delta, synapse=0.01)
		probe_memory = nengo.Probe(ens_memory, synapse=0.01)
		probe_weight = nengo.Probe(ens_weight, synapse=0.01)
		probe_project = nengo.Probe(ens_project, synapse=0.01)
		if train:
			probe_error = nengo.Probe(ens_error, synapse=0.01)
			probe_learned_weights = nengo.Probe(conn_weight, "weights")
		network.probe_target = probe_target
		network.probe_weight = probe_weight

	sim = nengo.Simulator(network, dt=0.001, progress_bar=False)
	with sim:
		sim.run(env.Tall)

	if plot:
		with sns.axes_style("whitegrid"):  # This enables gridlines
			fig, axes = plt.subplots(nrows=2, sharex=True, figsize=(8,6))
			sns.lineplot(x=sim.trange().flatten(), y=sim.data[probe_stim].flatten(), label='observations', color=palette[0], ax=axes[0])
			sns.lineplot(x=sim.trange().flatten(), y=sim.data[probe_memory].flatten(), label='memory', color=palette[0], ax=axes[0])
			if train:
				sns.lineplot(x=sim.trange().flatten(), y=sim.data[probe_error].flatten(), label='error', color=palette[1], ax=axes[0])
			sns.lineplot(x=sim.trange().flatten(), y=sim.data[probe_target].flatten(), label='target', color=palette[2], ax=axes[1])
			sns.lineplot(x=sim.trange().flatten(), y=sim.data[probe_weight].flatten(), label='weight', color=palette[3], ax=axes[1])
			# axes[0].set(title=f"training iteration {seed_stim}")
			axes[1].set(xlabel='time (s)')
			plt.grid(True, axis='y')  # Enable only y-axis gridlines
			fig.tight_layout()

	if train:
		learned_weights = sim.data[probe_learned_weights][-1]
		return learned_weights
	else:
		return network, sim

if __name__ == '__main__':
	dataset = sys.argv[1]
	sid = int(sys.argv[2])
	neurons = int(sys.argv[3])
	empirical = pd.read_pickle(f"data/{dataset}.pkl").query("sid==@sid")
	trials = empirical['trial'].unique()
	training_trials = np.arange(1, 41, 1)
	testing_trials = np.arange(1, 41, 1)
	rng = np.random.RandomState(seed=0)
	W = np.zeros((1, neurons))
	dfs = []
	columns = ['type', 'neurons', 'trial', 'time', 'target', 'error']
	start = time.time()
	for t in testing_trials:
		print(f"training iteration {t}")
		env = Environment(dataset, sid=sid, trial=t)
		W = run(W, env, n_neurons=neurons, seed_net=sid, train=True, plot=False)
	for t in testing_trials:
		print(f"testing iteration {t}")
		env = Environment(dataset, sid=sid, trial=t)
		net, sim = run(W, env, n_neurons=neurons, seed_net=sid, train=False, plot=False)
		times = sim.trange().flatten()
		targets = sim.data[net.probe_target].flatten()
		estimates = sim.data[net.probe_weight].flatten()
		errors = np.abs(targets - estimates)
		df = pd.DataFrame(columns=columns)
		df['type'] = ['test_type' for t in times]
		df['neurons'] = neurons
		df['trial'] = t
		df['time'] = times
		df['target'] = targets
		df['error'] = errors
		dfs.append(df)
	noise_data = pd.concat(dfs, ignore_index=True)
	noise_data.to_pickle(f"data/{dataset}_{sid}_{neurons}_counting.pkl")
	end = time.time()
	print(f"runtime {(end-start)/60:.4} min")