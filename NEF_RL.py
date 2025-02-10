import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import nengo
import pandas as pd

class Environment():
    def __init__(self, dataset, sid, trial, T=1, dt=0.001, dim_context=5, seed_env=0, decay='none', s=[1,1,1,1]):
        self.T = T
        self.dt = dt
        self.sid = sid
        self.trial = trial
        self.dataset = dataset
        self.empirical = pd.read_pickle(f"data/{dataset}.pkl").query("sid==@sid & trial==@trial")
        self.color = 0
        self.degree = 0
        self.stage = 0
        self.s = s  # learning rate multiplier for stages 0-3
        self.dim_context = dim_context
        self.decay = decay
        self.rng = np.random.RandomState(seed=seed_env)
        self.context = self.rng.rand(self.dim_context)
        self.context = self.context / np.linalg.norm(self.context)
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
        self.decays = []
        tt = int(self.T / self.dt)
        if self.dataset=='jiang':
            for stage in self.stages:
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
        if self.dataset=='carrabin':
            for stage in self.stages:
                color = self.empirical.query("stage==@stage")['color'].unique()[0]
                degree = 0
                decay = self.s[stage-1]
                self.colors.extend(color * np.ones((tt, 1)))
                self.degrees.extend(degree * np.ones((tt, 1)))
                self.decays.extend(decay * np.ones((tt, 1)))
        self.colors = np.array(self.colors).flatten()
        self.degrees = np.array(self.degrees).flatten()
        self.decays = np.array(self.decays).flatten()
                    
    def sample(self, t):
        tidx = int(t/self.dt)
        return [self.colors[tidx], self.degrees[tidx], self.decays[tidx]]

def build_network_RL(env, n_neurons=1000, seed_net=0, a=5e-5, z=0, syn=0.01):
    nengo.rc.set("decoder_cache", "enabled", "False")
    net = nengo.Network(seed=seed_net)
    net.z = z
    net.a = a
    net.syn = syn
    net.radius = 3 if env.dataset=='jiang' else env.s[0]
    net.radius2 = 3 if env.dataset=='jiang' else 1
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
        net.obs = nengo.Ensemble(n_neurons, 1)
        net.degree = nengo.Ensemble(n_neurons, 1)
        net.decay = nengo.Ensemble(n_neurons, 1, radius=net.radius)
        net.weight = nengo.Ensemble(n_neurons, 1, radius=net.radius)
        net.context = nengo.Ensemble(n_neurons, env.dim_context)
        net.prediction = nengo.Ensemble(n_neurons, 1)
        net.combined = nengo.Ensemble(n_neurons, 3, radius=net.radius2)
        net.error = nengo.Ensemble(n_neurons, 1)
        # connections
        nengo.Connection(net.input_obs, net.obs)
        nengo.Connection(net.input_degree, net.degree)
        nengo.Connection(net.input_context, net.context)
        nengo.Connection(net.input_decay, net.decay)
        nengo.Connection(net.degree, net.weight, transform=net.z, synapse=net.syn)
        nengo.Connection(net.decay, net.weight, synapse=net.syn)
        net.conn = nengo.Connection(net.context, net.prediction, learning_rule_type=net.pes, function=zero)
        nengo.Connection(net.obs, net.combined[0])
        nengo.Connection(net.prediction, net.combined[1])
        nengo.Connection(net.weight, net.combined[2], synapse=net.syn)
        nengo.Connection(net.combined, net.error, function=func_error, synapse=net.syn)
        nengo.Connection(net.error, net.conn.learning_rule)
        # probes
        net.probe_input_obs = nengo.Probe(net.input_obs, synapse=0)
        net.probe_input_degree = nengo.Probe(net.input_degree, synapse=0)
        net.probe_obs = nengo.Probe(net.obs, synapse=net.syn)
        net.probe_weight = nengo.Probe(net.weight, synapse=net.syn)
        net.probe_prediction = nengo.Probe(net.prediction, synapse=net.syn)
        net.probe_error = nengo.Probe(net.error, synapse=net.syn)
        net.probe_combined = nengo.Probe(net.combined, synapse=net.syn)
        net.probe_weight_neurons = nengo.Probe(net.weight.neurons, synapse=net.syn)
        net.probe_error_neurons = nengo.Probe(net.error.neurons, synapse=net.syn)
    return net

def simulate_RL(env, n_neurons=200, z=0, a=5e-5, seed_sim=0, seed_net=0, progress_bar=True):
    net = build_network_RL(env, n_neurons=n_neurons, seed_net=seed_net, z=z, a=a)
    sim = nengo.Simulator(net, seed=seed_sim, progress_bar=progress_bar)
    with sim:
        sim.run(env.Tall, progress_bar=progress_bar)
    return net, sim


def run_RL(dataset, sid, z, s=[1,1,1,1], a=5e-5, decay='stages', save=True):
    empirical = pd.read_pickle(f"data/{dataset}.pkl").query("sid==@sid")
    trials = empirical['trial'].unique() 
    columns = ['type', 'sid', 'trial', 'stage', 'estimate']
    dfs = []
    for trial in trials:
        print(f"sid {sid}, trial {trial}")
        env = Environment(dataset=dataset, sid=sid, trial=trial, decay=decay, s=s)
        seed_net = sid + 1000*trial
        net, sim = simulate_RL(env=env, seed_net=seed_net, z=z, a=a, progress_bar=False)
        n_observations = 0
        for stage in env.stages:
            subdata = empirical.query("trial==@trial and stage==@stage")
            if dataset=='jiang':
                observations = subdata['color'].to_numpy()
                for o in range(len(observations)):
                    n_observations += 1
                    tidx = int((n_observations*env.T)/env.dt)-2
                    estimate = sim.data[net.probe_prediction][tidx][0]
                    df = pd.DataFrame([['NEF_RL', sid, trial, stage, estimate]], columns=columns)
                    dfs.append(df)
            elif dataset=='carrabin':
                tidx = int((stage*env.T)/env.dt)-2
                estimate = sim.data[net.probe_prediction][tidx][0]
                df = pd.DataFrame([['NEF_RL', sid, trial, stage, estimate]], columns=columns)
                dfs.append(df)
    data = pd.concat(dfs, ignore_index=True)
    if save:
        data.to_pickle(f"data/NEF_RL_{dataset}_{sid}_estimates.pkl")
    return data

# def activity_RL(sid, z, s=[1,1,1,1], a=5e-5, decay='stages', save=True):
#     empirical = pd.read_pickle(f"data/human.pkl").query("sid==@sid")
#     trials = empirical['trial'].unique() 
#     columns = ['type', 'sid', 'trial', 'stage', 'tidx', 'aPE', 'RD', 'error activity', 'weight activity']
#     dfs = []
#     for trial in trials:
#         print(f"sid {sid}, trial {trial}")
#         env = Environment(sid=sid, trial=trial, decay=decay, s=s)
#         net, sim = simulate_RL(env=env, seed_net=sid, z=z, a=a, progress_bar=False)
#         n_observations = 0
#         for stage in range(4):
#             subdata = empirical.query("trial==@trial and stage==@stage")
#             observations = subdata['color'].to_numpy()
#             for o in range(len(observations)):
#                 tidx = int((n_observations*env.T)/env.dt)+200  # 200ms after stim presentation
#                 obs = sim.data[net.probe_input_obs][tidx][0]
#                 estimate = sim.data[net.probe_prediction][tidx][0]
#                 aPE = np.abs(obs - estimate)
#                 RD = sim.data[net.probe_input_degree][tidx][0]
#                 error_activity = np.mean(sim.data[net.probe_error_neurons][tidx])
#                 weight_activity = np.mean(sim.data[net.probe_weight_neurons][tidx])
#                 df = pd.DataFrame([['NEF_RL', sid, trial, stage, tidx, aPE, RD, error_activity, weight_activity]], columns=columns)
#                 dfs.append(df)
#                 n_observations += 1
#         data = pd.concat(dfs, ignore_index=True)
#         if save:
#             data.to_pickle(f"data/NEF_RL_{sid}_activities.pkl")
#     return data