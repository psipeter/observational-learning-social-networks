import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import nengo
import pandas as pd

class Environment():
    def __init__(self, sid, trial, time_sample=1, dt=0.001, dim_context=5, seed_env=0):
        self.time_sample = time_sample
        self.dt = dt
        self.sid = sid
        self.trial = trial
        self.empirical = pd.read_pickle(f"data/behavior.pkl").query("sid==@sid & trial==@trial")
        self.color = 0
        self.decay = 0
        self.degree = 0
        self.stage = 0
        self.dim_context = dim_context
        self.rng = np.random.RandomState(seed=seed_env)
        self.context = self.rng.rand(self.dim_context)
        self.context = self.context / np.linalg.norm(self.context)
        self.n_neighbors = len(self.empirical['who'].unique()) - 1
        # create input arrays
        self.colors = []
        self.decays = []
        self.degrees = []
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
            else:
                for n in range(self.n_neighbors):
                    color = self.empirical.query("stage==@stage")['color'].to_numpy()[n]
                    degree = 0 if stage==1 else self.empirical.query("stage==@stage")['RD'].to_numpy()[n]
                    n_samples = 1 + (stage-1)*self.n_neighbors + (n+1)                      
                    decay = 1 / n_samples
                    self.colors.extend(2*color-1 * np.ones((tt, 1)))
                    self.decays.extend(decay * np.ones((tt, 1)))
                    self.degrees.extend(degree * np.ones((tt, 1)))
        self.colors = np.array(self.colors).flatten()
        self.decays = np.array(self.decays).flatten()
        self.degrees = np.array(self.degrees).flatten()
                    
    def sample(self, t):
        tidx = int(t/self.dt)
        return [self.colors[tidx], self.decays[tidx], self.degrees[tidx]]

def build_network_RL(env, n_neurons=500, seed_net=0, syn_feedback=0.1, learning_rate=1e-4, z=0, k=1):
    nengo.rc.set("decoder_cache", "enabled", "False")
    net = nengo.Network(seed=seed_net)
    net.z = z
    net.k = k

    func_obs = lambda t: env.sample(t)[0]
    func_decay = lambda t: env.sample(t)[1]
    func_degree = lambda t: env.sample(t)[2]
    func_context = lambda t: env.context
    func_error = lambda x: x[2] * (x[0] - x[1])
    func_scale_decay = lambda x: np.abs(x[0])**net.k  # abs(x) avoids eval_points but with x<0
    func_choice = lambda x: -1 if x[0] < 0 else 1

    with net:
        # external inputs
        net.input_obs = nengo.Node(func_obs)
        net.input_decay = nengo.Node(func_decay)
        net.input_degree = nengo.Node(func_degree)
        net.input_context = nengo.Node(func_context)
        # ensembles
        net.obs = nengo.Ensemble(n_neurons, 1)
        net.decay = nengo.Ensemble(n_neurons, 1)
        net.degree = nengo.Ensemble(n_neurons, 1)
        net.weight = nengo.Ensemble(4*n_neurons, 1, radius=2)
        net.context = nengo.Ensemble(n_neurons*env.dim_context, env.dim_context)
        net.prediction = nengo.Ensemble(n_neurons, 1)
        net.combined = nengo.Ensemble(n_neurons*3, 3, radius=3)
        net.error = nengo.Ensemble(n_neurons, 1)
        net.decision = nengo.Ensemble(n_neurons, 1)
        # connections
        nengo.Connection(net.input_obs, net.obs)
        nengo.Connection(net.input_decay, net.decay)
        nengo.Connection(net.input_degree, net.degree)
        nengo.Connection(net.input_context, net.context)
        nengo.Connection(net.decay, net.weight, function=func_scale_decay)  # decay term
        nengo.Connection(net.degree, net.weight, transform=net.z)  # scaled connectivity term
        net.conn = nengo.Connection(net.context, net.prediction,
                                    learning_rule_type=nengo.PES(learning_rate=learning_rate), function=lambda x: 0)
        nengo.Connection(net.prediction, net.combined[0])
        nengo.Connection(net.obs, net.combined[1])
        nengo.Connection(net.weight, net.combined[2])
        nengo.Connection(net.combined, net.error, function=func_error)
        nengo.Connection(net.error, net.conn.learning_rule)
        nengo.Connection(net.prediction, net.decision, function=func_choice)
        # probes
        net.probe_input = nengo.Probe(net.input_obs, synapse=0)
        net.probe_obs = nengo.Probe(net.obs, synapse=0.01)
        net.probe_weight = nengo.Probe(net.weight, synapse=0.01)
        net.probe_prediction = nengo.Probe(net.prediction, synapse=0.01)
        net.probe_error = nengo.Probe(net.error, synapse=0.01)
        net.probe_decision = nengo.Probe(net.decision, synapse=0.01)
    return net

def simulate_RL(env, z=0, k=1, learning_rate=1e-4, seed_sim=0, seed_net=0, progress_bar=True):
    net = build_network_RL(env, seed_net=seed_net, z=z, k=k, learning_rate=learning_rate)
    sim = nengo.Simulator(net, seed=seed_sim, progress_bar=progress_bar)
    with sim:
        sim.run(env.T, progress_bar=progress_bar)
    return net, sim


def run_RL(sid, z, k, learning_rate=5e-5, save=True):
    empirical = pd.read_pickle(f"data/behavior.pkl").query("sid==@sid")
    trials = empirical['trial'].unique() 
    columns = ['type', 'sid', 'trial', 'stage', 'obs', 'RD', 'action', 'estimate', 'error', 'z', 'k']
    dfs = []
    for trial in trials:
        print(f"sid {sid}, trial {trial}")
        env = Environment(sid=sid, trial=trial)
        net, sim = simulate_RL(env=env, seed_net=sid, z=z, k=k,
            learning_rate=learning_rate, progress_bar=False)
        n_observations = 0
        for stage in range(4):
            subdata = empirical.query("trial==@trial and stage==@stage")
            action_emp = subdata['action'].to_numpy()[0]
            tidx = int((env.time_sample + stage*env.n_neighbors*env.time_sample)/env.dt)-2
            action_sim = sim.data[net.probe_decision][tidx][0]
            action_emp = 2*action_emp - 1  # converts [1,0] into [1,-1]
            action_sim = 1 if action_sim > 0 else -1  # turn real-value model decision (decoded from neural signal) into binary choice
            error = 1 if action_sim!=action_emp else 0
            observations = subdata['color'].to_numpy()
            RDs = subdata['RD'].to_numpy()
            for o in range(len(observations)):
                n_observations += 1
                obs = 2*observations[o] - 1
                RD = RDs[o]
                t0 = int((n_observations*env.time_sample)/env.dt)-100
                t1 = int((n_observations*env.time_sample)/env.dt)-2
                estimate = np.mean(sim.data[net.probe_prediction][t0:t1])
                df = pd.DataFrame([['human', sid, trial, stage, obs, RD, action_emp, None, None, None, None]], columns=columns)
                dfs.append(df)
                df = pd.DataFrame([['model-RL', sid, trial, stage, obs, RD, action_sim, estimate, error, z, k]], columns=columns)
                dfs.append(df)
        # export on the fly, to preserve partial data if remote job times out
        data = pd.concat(dfs, ignore_index=True)
        if save:
            data.to_pickle(f"data/rl_{sid}.pkl")
    # data = pd.concat(dfs, ignore_index=True)
    # data.to_pickle(f"data/rl_{sid}.pkl")
    return data