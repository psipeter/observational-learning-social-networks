import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import nengo
import pandas as pd

class EnvironmentRL():
    def __init__(self, dataset, sid, trial, alpha=0.2, z=0, lambd=0, T=1, dt=0.001, dim_context=5, seed_env=0):
        self.alpha = alpha
        self.lambd = lambd
        self.z = z
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
        if self.dataset=='jiang':
            self.n_neighbors = len(self.empirical['who'].unique()) - 1
            self.Tall = 3*self.T + 3*self.n_neighbors*self.T - self.dt
            self.stages = range(4)
        if self.dataset=='carrabin':
            self.Tall = 5*self.T - self.dt
            self.stages = range(1, 6)
        # create input arrays
        self.colors = []
        self.weights = []
        tt = int(self.T / self.dt)
        if self.dataset=='jiang':
            for stage in self.stages:
                if stage==0:
                    color = self.empirical.query("stage==@stage")['color'].to_numpy()[0]
                    weight = 1
                    self.colors.extend(color * np.ones((3*tt, 1)))
                    self.weights.extend(weight * np.ones((3*tt, 1)))
                if stage==1:
                    for n in range(self.n_neighbors):
                        samples = 2+n
                        color = self.empirical.query("stage==@stage")['color'].to_numpy()[n]
                        weight = self.alpha * np.power(samples, -self.lambd)
                        self.colors.extend(color * np.ones((tt, 1)))
                        self.weights.extend(weight * np.ones((tt, 1)))
                if stage>1:
                    for n in range(self.n_neighbors):
                        samples = 2+(stage-1)*self.n_neighbors + n
                        RD = self.empirical.query("stage==@stage")['RD'].to_numpy()[n]
                        color = self.empirical.query("stage==@stage")['color'].to_numpy()[n]
                        weight = self.alpha * np.power(samples, -self.lambd) + self.z*RD
                        self.colors.extend(color * np.ones((tt, 1)))
                        self.weights.extend(weight * np.ones((tt, 1)))
        if self.dataset=='carrabin':
            for stage in self.stages:
                color = self.empirical.query("stage==@stage")['color'].unique()[0]
                weight = self.alpha * np.power(stage+2, -self.lambd)
                self.colors.extend(color * np.ones((tt, 1)))
                self.weights.extend(weight * np.ones((tt, 1)))
        self.colors = np.array(self.colors).flatten()
        self.weights = np.array(self.weights).flatten()
                    
    def sample(self, t):
        tidx = int(t/self.dt)
        return [self.colors[tidx], self.weights[tidx]]

def build_network_RL(env, n_neurons=100, n_learning=100, n_error=100, seed_net=0, a=6e-5, syn=0.01):
    nengo.rc.set("decoder_cache", "enabled", "False")
    net = nengo.Network(seed=seed_net)
    func_obs = lambda t: env.sample(t)[0]
    func_weight = lambda t: env.sample(t)[1]
    func_context = lambda t: env.context
    zero = lambda x: 0

    with net:
        # external inputs
        input_obs = nengo.Node(func_obs)
        input_weight = nengo.Node(func_weight)
        input_context = nengo.Node(func_context)
        # ensembles
        obs = nengo.Ensemble(n_neurons, 1)
        weight = nengo.Ensemble(n_neurons, 1)
        context = nengo.Ensemble(n_learning, env.dim_context)
        value = nengo.Ensemble(n_learning, 1)
        error = nengo.networks.Product(n_error, 1)
        # connections
        nengo.Connection(input_obs, obs)
        nengo.Connection(input_context, context)
        nengo.Connection(input_weight, weight)
        C = nengo.Connection(context, value, learning_rule_type=nengo.PES(learning_rate=a), function=zero, synapse=syn)
        nengo.Connection(obs, error.input_a, synapse=syn)
        nengo.Connection(value, error.input_a, transform=-1, synapse=syn)
        nengo.Connection(weight, error.input_b, synapse=syn)
        nengo.Connection(error.output, C.learning_rule, synapse=syn, transform=-1)
        # probes
        net.probe_input_obs = nengo.Probe(input_obs, synapse=0)
        net.probe_input_weight = nengo.Probe(input_weight, synapse=0)
        net.probe_obs = nengo.Probe(obs, synapse=syn)
        net.probe_weight = nengo.Probe(weight, synapse=syn)
        net.probe_context = nengo.Probe(context, synapse=syn)
        net.probe_value = nengo.Probe(value, synapse=syn)
        net.probe_error = nengo.Probe(error.output, synapse=syn)
        net.probe_obs_spikes = nengo.Probe(obs.neurons, synapse=None)
        net.probe_weight_spikes = nengo.Probe(weight.neurons, synapse=None)
        net.probe_context_spikes = nengo.Probe(context.neurons, synapse=None)
        net.probe_value_spikes = nengo.Probe(value.neurons, synapse=None)
        net.probe_error1_spikes = nengo.Probe(error.sq1.ea_ensembles[0].neurons, synapse=None)
        net.probe_error2_spikes = nengo.Probe(error.sq2.ea_ensembles[0].neurons, synapse=None)

    return net

def simulate_RL(env, n_neurons=100, n_learning=100, n_error=100, seed_sim=0, seed_net=0, progress_bar=True):
    net = build_network_RL(env, n_neurons=n_neurons, n_learning=n_learning, n_error=n_error, seed_net=seed_net)
    sim = nengo.Simulator(net, seed=seed_sim, progress_bar=progress_bar)
    with sim:
        sim.run(env.Tall, progress_bar=progress_bar)
    return net, sim

def run_RL(dataset, sid, alpha, z, lambd, n_neurons=100, n_learning=100, n_error=100):
    empirical = pd.read_pickle(f"data/{dataset}.pkl").query("sid==@sid")
    trials = empirical['trial'].unique() 
    columns = ['type', 'sid', 'trial', 'stage', 'estimate']
    dfs = []
    for trial in trials:
        print(f"sid {sid}, trial {trial}")
        env = EnvironmentRL(dataset=dataset, sid=sid, trial=trial, alpha=alpha, z=z, lambd=lambd)
        seed_net = sid + 1000*trial
        net, sim = simulate_RL(env=env, n_neurons=n_neurons, n_learning=n_learning, n_error=n_error, seed_net=seed_net, progress_bar=False)
        if dataset=='jiang':
            obs_times = np.arange(3, 3+4*env.n_neighbors, env.n_neighbors) * env.T/env.dt
        elif dataset=='carrabin':
            obs_times = np.arange(1, 6, 1) * env.T/env.dt
        obs_times = obs_times.astype(int)
        for s, tidx in enumerate(obs_times):
            stage = env.stages[s]
            estimate = np.mean(sim.data[net.probe_value][tidx-100: tidx])
            dfs.append(pd.DataFrame([['NEF_RL', sid, trial, stage, estimate]], columns=columns))
    data = pd.concat(dfs, ignore_index=True)
    data.to_pickle(f"data/NEF_RL_{dataset}_{sid}_estimates.pkl")
    return data