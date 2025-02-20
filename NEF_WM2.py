import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import nengo
import pandas as pd

class EnvironmentWM():
    def __init__(self, dataset, sid, trial, T=1, dt=0.001):
        self.T = T
        self.dt = dt
        self.sid = sid
        self.trial = trial
        self.dataset = dataset
        self.empirical = pd.read_pickle(f"data/{dataset}.pkl").query("sid==@sid & trial==@trial")
        self.color = 0
        self.degree = 0
        self.stage = 0
        if self.dataset=='jiang':
            self.n_neighbors = len(self.empirical['who'].unique()) - 1
            self.Tall = self.T + 3*self.n_neighbors*self.T - self.dt
            self.stages = range(4)
        if self.dataset=='carrabin':
            self.Tall = 5*self.T - self.dt
            self.stages = range(1, 6)
        # create input arrays
        self.colors = []
        self.degrees = []
        tt = int(self.T / self.dt)
        if self.dataset=='jiang':
            for stage in self.stages:
                if stage==0:
                    color = self.empirical.query("stage==@stage")['color'].to_numpy()[0]
                    degree = 0
                    self.colors.extend(color * np.ones((tt, 1)))
                    self.degrees.extend(degree * np.ones((tt, 1)))
                else:
                    for n in range(self.n_neighbors):
                        color = self.empirical.query("stage==@stage")['color'].to_numpy()[n]
                        degree = 0 if stage==1 else self.empirical.query("stage==@stage")['RD'].to_numpy()[n]
                        self.colors.extend(color * np.ones((tt, 1)))
                        self.degrees.extend(degree * np.ones((tt, 1)))
        if self.dataset=='carrabin':
            for stage in self.stages:
                color = self.empirical.query("stage==@stage")['color'].unique()[0]
                self.colors.extend(color * np.ones((tt, 1)))
                self.degrees.extend(0 * np.ones((tt, 1)))
        self.colors = np.array(self.colors).flatten()
        self.degrees = np.array(self.degrees).flatten()
                    
    def sample(self, t):
        tidx = int(t/self.dt)
        return [self.colors[tidx], self.degrees[tidx]]

def build_network_WM(env, alpha=0.2, a=5e-5, z=0, n_neurons=100, n_memory=100, n_error=100, seed_net=0, syn_ff=0.01, syn_fb=0.1, w_ff=0.1):

    nengo.rc.set("decoder_cache", "enabled", "False")
    net = nengo.Network(seed=seed_net)
    func_obs = lambda t: env.sample(t)[0]
    func_degree = lambda t: env.sample(t)[1]
    func_alpha = lambda t: alpha

    with net:
        # external inputs
        input_obs = nengo.Node(func_obs)
        input_degree = nengo.Node(func_degree)
        input_alpha = nengo.Node(func_alpha)
        # ensembles
        obs = nengo.Ensemble(n_neurons, 1)
        weight = nengo.Ensemble(n_neurons, 1)
        value = nengo.Ensemble(n_memory, 1)
        error = nengo.networks.Product(n_error, 1)
        # connections
        nengo.Connection(input_obs, obs)
        nengo.Connection(input_degree, weight, transform=z)
        nengo.Connection(input_alpha, weight)
        nengo.Connection(obs, error.input_a, synapse=syn_ff)
        nengo.Connection(value, error.input_a, synapse=syn_ff, transform=-1)
        nengo.Connection(weight, error.input_b, synapse=syn_ff)
        nengo.Connection(error.output, value, synapse=syn_ff, transform=w_ff)
        nengo.Connection(value, value, synapse=syn_fb)
        # probes
        net.probe_input_obs = nengo.Probe(input_obs, synapse=0)
        net.probe_input_degree = nengo.Probe(input_degree, synapse=0)
        net.probe_obs = nengo.Probe(obs, synapse=syn_ff)
        net.probe_weight = nengo.Probe(weight, synapse=syn_ff)
        net.probe_value = nengo.Probe(value, synapse=syn_ff)
        net.probe_error = nengo.Probe(error.output, synapse=syn_ff)

    return net

def simulate_WM(env, alpha=0.2, z=0, n_neurons=100, n_memory=100, n_error=100, seed_sim=0, seed_net=0, progress_bar=True):
    net = build_network_WM(env, alpha=alpha, z=z, n_neurons=n_neurons, n_memory=n_memory, n_error=n_error, seed_net=seed_net)
    sim = nengo.Simulator(net, seed=seed_sim, progress_bar=progress_bar)
    with sim:
        sim.run(env.Tall, progress_bar=progress_bar)
    return net, sim

def run_WM(dataset, sid, alpha, z, n_neurons=100, n_memory=100, n_error=100, save=True):
    empirical = pd.read_pickle(f"data/{dataset}.pkl").query("sid==@sid")
    trials = empirical['trial'].unique() 
    columns = ['type', 'sid', 'trial', 'stage', 'estimate']
    dfs = []
    for trial in trials:
        print(f"sid {sid}, trial {trial}")
        env = EnvironmentWM(dataset=dataset, sid=sid, trial=trial)
        seed_net = sid + 1000*trial
        net, sim = simulate_WM(env=env, alpha=alpha, z=z, n_neurons=n_neurons, n_memory=n_memory, n_error=n_error, seed_net=seed_net, progress_bar=False)
        n_observations = 0
        for stage in env.stages:
            subdata = empirical.query("trial==@trial and stage==@stage")
            if dataset=='jiang':
                observations = subdata['color'].to_numpy()
                for o in range(len(observations)):
                    n_observations += 1
                    tidx = int((n_observations*env.T)/env.dt)-2
                    estimate = np.mean(sim.data[net.probe_value][tidx-100: tidx])
                    df = pd.DataFrame([['NEF_WM', sid, trial, stage, estimate]], columns=columns)
                    dfs.append(df)
            elif dataset=='carrabin':
                tidx = int((stage*env.T)/env.dt)-2
                estimate = np.mean(sim.data[net.probe_value][tidx-100: tidx])
                df = pd.DataFrame([['NEF_WM', sid, trial, stage, estimate]], columns=columns)
                dfs.append(df)
    data = pd.concat(dfs, ignore_index=True)
    if save:
        data.to_pickle(f"data/NEF_WM_{dataset}_{sid}_estimates.pkl")
    return data