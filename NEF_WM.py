import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import nengo
import pandas as pd

class EnvironmentWM():
    def __init__(self, dataset, sid, trial, alpha=0.2, z=0, lambd=0, T=1, dt=0.001):
        self.alpha = alpha
        self.lambd = lambd
        self.z = z
        self.T = T
        self.dt = dt
        self.sid = sid
        self.trial = trial
        self.dataset = dataset
        self.empirical = pd.read_pickle(f"data/{dataset}.pkl").query("sid==@sid & trial==@trial")
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

def build_network_WM(env, a=5e-5, n_neurons=100, n_memory=100, n_error=100, seed_net=0, syn_ff=0.01, syn_fb=0.1, w_ff=0.1):

    nengo.rc.set("decoder_cache", "enabled", "False")
    net = nengo.Network(seed=seed_net)
    func_obs = lambda t: env.sample(t)[0]
    func_weight = lambda t: env.sample(t)[1]

    with net:
        # external inputs
        input_obs = nengo.Node(func_obs)
        input_weight = nengo.Node(func_weight)
        # ensembles
        obs = nengo.Ensemble(n_neurons, 1)
        weight = nengo.Ensemble(n_neurons, 1)
        value = nengo.Ensemble(n_memory, 1)
        error = nengo.networks.Product(n_error, 1)
        # connections
        nengo.Connection(input_obs, obs)
        nengo.Connection(input_weight, weight)
        nengo.Connection(obs, error.input_a, synapse=syn_ff)
        nengo.Connection(value, error.input_a, synapse=syn_ff, transform=-1)
        nengo.Connection(weight, error.input_b, synapse=syn_ff)
        nengo.Connection(error.output, value, synapse=syn_ff, transform=w_ff)
        nengo.Connection(value, value, synapse=syn_fb)
        # probes
        net.probe_input_obs = nengo.Probe(input_obs, synapse=0)
        net.probe_input_weight = nengo.Probe(input_weight, synapse=0)
        net.probe_obs = nengo.Probe(obs, synapse=syn_ff)
        net.probe_weight = nengo.Probe(weight, synapse=syn_ff)
        net.probe_value = nengo.Probe(value, synapse=syn_ff)
        net.probe_error = nengo.Probe(error.output, synapse=syn_ff)

    return net

def simulate_WM(env, n_neurons=100, n_memory=100, n_error=100, seed_sim=0, seed_net=0, progress_bar=True):
    net = build_network_WM(env, n_neurons=n_neurons, n_memory=n_memory, n_error=n_error, seed_net=seed_net)
    sim = nengo.Simulator(net, seed=seed_sim, progress_bar=progress_bar)
    with sim:
        sim.run(env.Tall, progress_bar=progress_bar)
    return net, sim

def run_WM(dataset, sid, alpha, z, lambd, n_neurons=100, n_memory=100, n_error=100, save=True):
    empirical = pd.read_pickle(f"data/{dataset}.pkl").query("sid==@sid")
    trials = empirical['trial'].unique() 
    columns = ['type', 'sid', 'trial', 'stage', 'estimate']
    dfs = []
    for trial in trials:
        print(f"sid {sid}, trial {trial}")
        env = EnvironmentWM(dataset=dataset, sid=sid, trial=trial, alpha=alpha, z=z, lambd=lambd)
        seed_net = sid + 1000*trial
        net, sim = simulate_WM(env=env, n_neurons=n_neurons, n_memory=n_memory, n_error=n_error, seed_net=seed_net, progress_bar=False)
        if dataset=='jiang':
            obs_times = np.arange(3, 3+4*env.n_neighbors, env.n_neighbors) * env.T/env.dt
        elif dataset=='carrabin':
            obs_times = np.arange(1, 6, 1) * env.T/env.dt
        obs_times = obs_times.astype(int)
        for s, tidx in enumerate(obs_times):
            stage = env.stages[s]
            estimate = np.mean(sim.data[net.probe_value][tidx-100: tidx])
            dfs.append(pd.DataFrame([['NEF_WM', sid, trial, stage, estimate]], columns=columns))
    data = pd.concat(dfs, ignore_index=True)
    if save:
        data.to_pickle(f"data/NEF_WM_{dataset}_{sid}_estimates.pkl")
    return data