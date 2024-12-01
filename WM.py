import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import nengo
import pandas as pd

class Environment():
    def __init__(self, sid, trial, time_sample=2, dt=0.001):
        self.time_sample = time_sample
        self.dt = dt
        self.sid = sid
        self.trial = trial
        self.empirical = pd.read_pickle(f"data/behavior.pkl").query("sid==@sid & trial==@trial")
        self.color = 0
        self.decay = 0
        self.degree = 0
        self.inhibit1 = 0
        self.inhibit2 = 0
        self.stage = 0
        self.n_neighbors = len(self.empirical['who'].unique()) - 1
        # create input arrays
        self.colors = []
        self.decays = []
        self.degrees = []
        self.inhibit1s = []
        self.inhibit2s = []
        tt = int(self.time_sample / self.dt)
        tt2 = int(tt/2)
        self.T = self.time_sample + 3*self.n_neighbors*self.time_sample - self.dt
        for stage in range(4):
            if stage==0:
                color = self.empirical.query("stage==@stage")['color'].to_numpy()[0]
                n_samples = 1
                decay = 1
                degree = 0
                self.colors.extend(2*color-1 * np.ones((tt, 1)))
                self.decays.extend(decay * np.ones((tt, 1)))
                self.degrees.extend(degree * np.ones((tt, 1)))
                self.inhibit1s.extend(1 * np.ones((tt2, 1)))
                self.inhibit2s.extend(0 * np.ones((tt2, 1)))
                self.inhibit1s.extend(0 * np.ones((tt2, 1)))
                self.inhibit2s.extend(1 * np.ones((tt2, 1)))
            else:
                for n in range(self.n_neighbors):
                    color = self.empirical.query("stage==@stage")['color'].to_numpy()[n]
                    degree = 0 if stage==1 else self.empirical.query("stage==@stage")['RD'].to_numpy()[n]
                    n_samples = 1 + (stage-1)*self.n_neighbors + (n+1)                      
                    decay = 1 / n_samples
                    self.colors.extend(2*color-1 * np.ones((tt, 1)))
                    self.decays.extend(decay * np.ones((tt, 1)))
                    self.degrees.extend(degree * np.ones((tt, 1)))
                    self.inhibit1s.extend(1 * np.ones((tt2, 1)))
                    self.inhibit2s.extend(0 * np.ones((tt2, 1)))
                    self.inhibit1s.extend(0 * np.ones((tt2, 1)))
                    self.inhibit2s.extend(1 * np.ones((tt2, 1)))
        self.colors = np.array(self.colors).flatten()
        self.decays = np.array(self.decays).flatten()
        self.degrees = np.array(self.degrees).flatten()
        self.inhibit1s = np.array(self.inhibit1s).flatten()
        self.inhibit2s = np.array(self.inhibit2s).flatten()
                    
    def sample(self, t):
        tidx = int(t/self.dt)
        return [self.colors[tidx], self.decays[tidx], self.degrees[tidx], self.inhibit1s[tidx], self.inhibit2s[tidx]]


def build_network_WM(env, n_neurons=500, seed_net=0, syn_feedback=0.1, z=0, k=1):
    nengo.rc.set("decoder_cache", "enabled", "False")
    net = nengo.Network(seed=seed_net)
    net.z = z
    net.k = k
    func_obs = lambda t: env.sample(t)[0]
    func_decay = lambda t: env.sample(t)[1]
    func_degree = lambda t: env.sample(t)[2]
    func_inh1 = lambda t: env.sample(t)[3]
    func_inh2 = lambda t: env.sample(t)[4]
    func_load = lambda x: (x[0] - x[1]) * x[2]
    func_diff = lambda x: (x[0] - x[1])
    func_scale_decay = lambda x: np.abs(x[0])**net.k  # abs(x) avoids eval_points but with x<0
    func_choice = lambda x: -1 if x[0] < 0 else 1

    with net:
        # external inputs
        net.input_obs = nengo.Node(func_obs)
        net.input_decay = nengo.Node(func_decay)
        net.input_degree = nengo.Node(func_degree)
        net.inhibit1 = nengo.Node(func_inh1)
        net.inhibit2 = nengo.Node(func_inh2)
        net.weight_out = nengo.Node(size_in=1)
        # ensembles
        net.obs = nengo.Ensemble(n_neurons, 1)
        net.decay = nengo.Ensemble(n_neurons, 1)
        net.degree = nengo.Ensemble(n_neurons, 1)
        net.weight = nengo.Ensemble(4*n_neurons, 1, radius=2)
        net.combined = nengo.Ensemble(4*n_neurons, 4, radius=3)
        net.memory = nengo.Ensemble(n_neurons, 1)
        net.temp = nengo.Ensemble(n_neurons, 2)
        net.old = nengo.Ensemble(n_neurons, 1)
        net.decision = nengo.Ensemble(n_neurons, 1)
        # connections
        nengo.Connection(net.input_obs, net.obs)
        nengo.Connection(net.input_decay, net.decay)
        nengo.Connection(net.input_degree, net.degree)
        nengo.Connection(net.decay, net.weight, function=func_scale_decay)  # decay term
        nengo.Connection(net.degree, net.weight, transform=net.z)  # scaled connectivity term
        nengo.Connection(net.obs, net.combined[0])
        nengo.Connection(net.old, net.combined[1])
        nengo.Connection(net.weight, net.combined[2])
        nengo.Connection(net.combined, net.memory, function=func_load, transform=syn_feedback)
        nengo.Connection(net.memory, net.memory, synapse=syn_feedback)
        nengo.Connection(net.memory, net.temp[0])
        nengo.Connection(net.old, net.temp[1])
        nengo.Connection(net.temp, net.old, function=func_diff)
        nengo.Connection(net.old, net.old, synapse=syn_feedback)
        nengo.Connection(net.inhibit1, net.temp.neurons, transform=-1e3*np.ones((net.temp.n_neurons, 1)))
        nengo.Connection(net.inhibit2, net.combined.neurons, transform=-1e3*np.ones((net.combined.n_neurons, 1)))
        nengo.Connection(net.memory, net.decision, function=func_choice)
        # probes
        net.probe_input = nengo.Probe(net.input_obs, synapse=0.01)
        net.probe_obs = nengo.Probe(net.obs, synapse=0.01)
        net.probe_weight = nengo.Probe(net.weight, synapse=0.01)
        net.probe_memory = nengo.Probe(net.memory, synapse=0.01)
        net.probe_old = nengo.Probe(net.old, synapse=0.01)
        net.probe_decision = nengo.Probe(net.decision, synapse=0.01)
    return net

def simulate_WM(env, z=0, k=1, seed_sim=0, seed_net=0, progress_bar=True):
    net = build_network_WM(env, seed_net=seed_net, z=z, k=k)
    sim = nengo.Simulator(net, seed=seed_sim, progress_bar=progress_bar)
    with sim:
        sim.run(env.T, progress_bar=progress_bar)
    return net, sim

def run_WM(sid, z, k, save=True):
    empirical = pd.read_pickle(f"data/behavior.pkl").query("sid==@sid")
    trials = empirical['trial'].unique()
    columns = ['type', 'sid', 'trial', 'stage', 'estimate']
    dfs = []
    for trial in trials:
        print(f"sid {sid}, trial {trial}")
        env = Environment(sid=sid, trial=trial)
        net, sim = simulate_WM(env=env, seed_net=sid, z=z, k=k, progress_bar=False)
        n_observations = 0
        for stage in range(4):
            subdata = empirical.query("trial==@trial and stage==@stage")
            observations = subdata['color'].to_numpy()
            RDs = subdata['RD'].to_numpy()
            for o in range(len(observations)):
                n_observations += 1
                t0 = int((n_observations*env.time_sample)/env.dt)-100
                t1 = int((n_observations*env.time_sample)/env.dt)-2
                estimate = np.mean(sim.data[net.probe_memory][t0:t1])
                df = pd.DataFrame([['WM', sid, trial, stage, estimate]], columns=columns)
                dfs.append(df)
        data = pd.concat(dfs, ignore_index=True)
        if save:
            data.to_pickle(f"data/WM_{sid}.pkl")
    return data