import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import nengo
import pandas as pd

class Environment():
    def __init__(self, sid, trial, time_sample=1, dt=0.001, dim_context=5, seed_env=0,
                 decay='none', s=[1,1,1,1]):
        self.time_sample = time_sample
        self.dt = dt
        self.sid = sid
        self.trial = trial
        self.empirical = pd.read_pickle(f"data/human.pkl").query("sid==@sid & trial==@trial")
        self.color = 0
        self.degree = 0
        self.stage = 0
        self.s = s  # learning rate multiplier for stages 0-3
        self.dim_context = dim_context
        self.decay = decay
        self.rng = np.random.RandomState(seed=seed_env)
        self.context = self.rng.rand(self.dim_context)
        self.context = self.context / np.linalg.norm(self.context)
        self.n_neighbors = len(self.empirical['who'].unique()) - 1
        # create input arrays
        self.colors = []
        self.degrees = []
        self.decays = []
        tt = int(self.time_sample / self.dt)
        self.T = self.time_sample + 3*self.n_neighbors*self.time_sample - self.dt
        for stage in range(4):
            if stage==0:
                color = self.empirical.query("stage==@stage")['color'].to_numpy()[0]
                degree = 0
                if self.decay=='stages':
                    decay = self.s[0]
                else:
                    decay = 1
                self.colors.extend(color * np.ones((tt, 1)))
                self.degrees.extend(degree * np.ones((tt, 1)))
                self.decays.extend(decay * np.ones((tt, 1)))
            else:
                for n in range(self.n_neighbors):
                    color = self.empirical.query("stage==@stage")['color'].to_numpy()[n]
                    degree = 0 if stage==1 else self.empirical.query("stage==@stage")['RD'].to_numpy()[n]
                    if self.decay=='none':
                        decay = 1
                    if self.decay=='stages':
                        decay = self.s[stage]
                    if self.decay=='samples':
                        n_samples = 1 + (stage-1)*self.n_neighbors + (n+1)                      
                        decay = 1 / n_samples
                    self.colors.extend(color * np.ones((tt, 1)))
                    self.degrees.extend(degree * np.ones((tt, 1)))
                    self.decays.extend(decay * np.ones((tt, 1)))
        self.colors = np.array(self.colors).flatten()
        self.degrees = np.array(self.degrees).flatten()
        self.decays = np.array(self.decays).flatten()
                    
    def sample(self, t):
        tidx = int(t/self.dt)
        return [self.colors[tidx], self.degrees[tidx], self.decays[tidx]]

def build_network_RL(env, n_neurons=1000, seed_net=0, a=1e-4, z=0, direct=False):
    nengo.rc.set("decoder_cache", "enabled", "False")
    net = nengo.Network(seed=seed_net)
    net.z = z
    net.a = a
    net.pes = nengo.PES(learning_rate=net.a)
    zero = lambda x: 0

    func_obs = lambda t: env.sample(t)[0]
    func_degree = lambda t: env.sample(t)[1]
    func_decay = lambda t: env.sample(t)[2]
    func_context = lambda t: env.context
    func_error = lambda x: -x[2] * (x[0] - x[1])

    with net:
        # external inputs
        net.input_obs = nengo.Node(func_obs)
        net.input_degree = nengo.Node(func_degree)
        net.input_context = nengo.Node(func_context)
        net.input_decay = nengo.Node(func_decay)
        # ensembles
        if direct:
            net.obs = nengo.Ensemble(1, 1, neuron_type=nengo.Direct())
            net.degree = nengo.Ensemble(1, 1, neuron_type=nengo.Direct())
            net.decay = nengo.Ensemble(1, 1, neuron_type=nengo.Direct())
            net.weight = nengo.Ensemble(1, 1, neuron_type=nengo.Direct())
            net.context = nengo.Ensemble(100, env.dim_context, neuron_type=nengo.LIF())  # must be neurons to have decoders
            net.prediction = nengo.Ensemble(1, 1, neuron_type=nengo.Direct())
            net.combined = nengo.Ensemble(1, 3, neuron_type=nengo.Direct())
            net.error = nengo.Ensemble(1, 1, neuron_type=nengo.Direct())
        else:
            net.obs = nengo.Ensemble(n_neurons, 1)
            net.degree = nengo.Ensemble(n_neurons, 1)
            net.decay = nengo.Ensemble(3*n_neurons, 1, radius=3)
            net.weight = nengo.Ensemble(3*n_neurons, 1, radius=3)
            net.context = nengo.Ensemble(n_neurons*env.dim_context, env.dim_context)
            net.prediction = nengo.Ensemble(n_neurons, 1)
            net.combined = nengo.Ensemble(3*n_neurons, 3, radius=3)
            net.error = nengo.Ensemble(n_neurons, 1)
        # connections
        nengo.Connection(net.input_obs, net.obs)
        nengo.Connection(net.input_degree, net.degree)
        nengo.Connection(net.input_context, net.context)
        nengo.Connection(net.input_decay, net.decay)
        nengo.Connection(net.degree, net.weight, transform=net.z, synapse=0.03)
        nengo.Connection(net.decay, net.weight, synapse=0.03)
        net.conn = nengo.Connection(net.context, net.prediction, learning_rule_type=net.pes, function=zero)
        nengo.Connection(net.obs, net.combined[0])
        nengo.Connection(net.prediction, net.combined[1])
        nengo.Connection(net.weight, net.combined[2], synapse=0.03)
        nengo.Connection(net.combined, net.error, function=func_error, synapse=0.03)
        nengo.Connection(net.error, net.conn.learning_rule)
        # probes
        net.probe_input = nengo.Probe(net.input_obs, synapse=0)
        net.probe_obs = nengo.Probe(net.obs, synapse=0.03)
        net.probe_weight = nengo.Probe(net.weight, synapse=0.03)
        net.probe_prediction = nengo.Probe(net.prediction, synapse=0.03)
        net.probe_error = nengo.Probe(net.error, synapse=0.03)
        net.probe_combined = nengo.Probe(net.combined, synapse=0.03)
    return net

def simulate_RL(env, z=0, a=1e-4, seed_sim=0, seed_net=0, progress_bar=True, direct=False):
    net = build_network_RL(env, seed_net=seed_net, z=z, a=a, direct=direct)
    sim = nengo.Simulator(net, seed=seed_sim, progress_bar=progress_bar)
    with sim:
        sim.run(env.T, progress_bar=progress_bar)
    return net, sim


def run_RL(sid, z, s=[1,1,1,1], a=5e-5, decay='stage', save=True, direct=False):
    empirical = pd.read_pickle(f"data/human.pkl").query("sid==@sid")
    trials = empirical['trial'].unique() 
    columns = ['type', 'sid', 'trial', 'stage', 'estimate']
    dfs = []
    for trial in trials:
        print(f"sid {sid}, trial {trial}")
        env = Environment(sid=sid, trial=trial, decay=decay, s=s)
        net, sim = simulate_RL(env=env, seed_net=sid, z=z, a=a, progress_bar=False, direct=direct)
        n_observations = 0
        for stage in range(4):
            subdata = empirical.query("trial==@trial and stage==@stage")
            observations = subdata['color'].to_numpy()
            for o in range(len(observations)):
                n_observations += 1
                tidx = int((n_observations*env.time_sample)/env.dt)-2
                estimate = sim.data[net.probe_prediction][tidx][0]
                df = pd.DataFrame([['NEF_RL', sid, trial, stage, estimate]], columns=columns)
                dfs.append(df)
        data = pd.concat(dfs, ignore_index=True)
        if save:
            data.to_pickle(f"data/NEF_RL_{sid}.pkl")
    return data