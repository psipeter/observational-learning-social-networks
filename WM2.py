import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import nengo
import pandas as pd

class Environment():
    def __init__(self, sid, trial, t_show=1.0, t_buffer=1.0, dt=0.001):
        self.dt = dt
        self.sid = sid
        self.trial = trial
        self.empirical = pd.read_pickle(f"data/human.pkl").query("sid==@sid & trial==@trial")
        self.t_show = t_show
        self.t_buffer = t_buffer
        self.T = self.t_show + self.t_buffer
        self.color = 0
        self.degree = 0
        self.stage = 0
        self.n_neighbors = len(self.empirical['who'].unique()) - 1
        self.Tall = self.T + 3*self.n_neighbors*self.T - self.dt
        # create input arrays
        self.colors = []
        self.degrees = []
        tt1 = int(self.t_show / self.dt)
        tt2 = int(self.t_buffer / self.dt)
        for stage in range(4):
            if stage==0:
                color = self.empirical.query("stage==@stage")['color'].to_numpy()[0]
                degree = 0
                self.colors.extend(color * np.ones((tt1, 1)))
                self.colors.extend(np.zeros((tt2, 1)))
                self.degrees.extend(degree * np.ones((tt1, 1)))
                self.degrees.extend(np.zeros((tt2, 1)))
            else:
                for n in range(self.n_neighbors):
                    color = self.empirical.query("stage==@stage")['color'].to_numpy()[n]
                    degree = 0 if stage==1 else self.empirical.query("stage==@stage")['RD'].to_numpy()[n]
                    self.colors.extend(color * np.ones((tt1, 1)))
                    self.colors.extend(np.zeros((tt2, 1)))
                    self.degrees.extend(degree * np.ones((tt1, 1)))
                    self.degrees.extend(np.zeros((tt2, 1)))
        self.colors = np.array(self.colors).flatten()
        self.degrees = np.array(self.degrees).flatten()
                    
    def sample(self, t):
        tidx = int(t/self.dt)
        return [self.colors[tidx], self.degrees[tidx]]


def build_network_WM(env, n_neurons=1000, seed_net=0, z=0, k=1):
    nengo.rc.set("decoder_cache", "enabled", "False")
    net = nengo.Network(seed=seed_net)
    net.z = z
    net.k = k
    net.encoders = nengo.dists.Choice([[1]])
    net.intercepts = nengo.dists.Uniform(0, 1)
    net.eval_points = np.linspace(-1, 1, 1000).reshape(-1,1)

    func_obs = lambda t: env.sample(t)[0]
    func_degree = lambda t: env.sample(t)[1]
    func_multiply = lambda x: x[0]*x[1]
    func_inverse = lambda x: 1/(x+10)
    func_center = lambda x: x-10
    func_stop2 = lambda x: 0 if np.abs(x)>0.1 else 1
    func_stop3 = lambda x: 1 if np.abs(x)>0.1 else 0
    func_differentiate1 = lambda x: 2*np.abs(x)
    func_differentiate2 = lambda x: -2*np.abs(x)
    func_power = lambda x: np.abs(x)**net.k

    w_stop = -10*np.ones((n_neurons, 1))

    with net:
        # inputs
        node_stim = nengo.Node(func_obs)
        node_degree = nengo.Node(func_degree)
        
        # neural populations
        ens_stim = nengo.Ensemble(n_neurons, 1)
        ens_differentiator = nengo.Ensemble(n_neurons, 1, encoders=net.encoders, intercepts=net.intercepts)
        ens_number_memory = nengo.Ensemble(2*n_neurons, 1)
        ens_number = nengo.Ensemble(n_neurons, 1, radius=20)
        ens_centered = nengo.Ensemble(n_neurons, 1, radius=11)
        ens_weight = nengo.Ensemble(n_neurons, 1, radius=0.9)
        ens_scaled_weight = nengo.Ensemble(n_neurons, 1)
        ens_degree = nengo.Ensemble(n_neurons, 1)
        ens_d1 = nengo.Ensemble(2*n_neurons, 2, radius=3)
        ens_d2 = nengo.Ensemble(n_neurons, 1)
        ens_d3 = nengo.Ensemble(n_neurons, 1)
        ens_memory = nengo.Ensemble(n_neurons, 1)
        ens_old = nengo.Ensemble(n_neurons, 1)
        ens_stop2 = nengo.Ensemble(n_neurons, 1, encoders=net.encoders, intercepts=net.intercepts)
        ens_stop3 = nengo.Ensemble(n_neurons, 1, encoders=net.encoders, intercepts=net.intercepts)

        # input signals
        nengo.Connection(node_stim, ens_stim, synapse=None)
        nengo.Connection(node_degree, ens_degree, synapse=None)
        
        # count up based on changes in the input signals
        nengo.Connection(ens_stim, ens_differentiator, synapse=0.01, function=func_differentiate1)
        nengo.Connection(ens_stim, ens_differentiator, synapse=0.1, function=func_differentiate2)
        nengo.Connection(ens_differentiator, ens_number_memory, synapse=0.2, transform=0.05)
        nengo.Connection(ens_number_memory, ens_number_memory, synapse=0.2)
        nengo.Connection(ens_number_memory, ens_number, transform=25, synapse=0.1)

        # compute weight = 1/N**k from number memory
        nengo.Connection(ens_number, ens_centered, synapse=0.01, function=func_center)
        nengo.Connection(ens_centered, ens_weight, synapse=0.03, function=func_inverse, eval_points=net.eval_points)
        nengo.Connection(ens_weight, ens_scaled_weight, synapse=0.03, function=func_power)

        # compute error between current observation and WM value stored in "old" buffer  
        # then pass that error, times the current weight, to D2
        # also add degree into current weight, as represented in D1
        nengo.Connection(ens_stim, ens_d1[0], synapse=0.01)
        nengo.Connection(ens_scaled_weight, ens_d1[1], synapse=0.03)
        nengo.Connection(ens_d1, ens_d2, synapse=0.01, function=func_multiply)
        nengo.Connection(ens_old, ens_d1[0], synapse=0.01, transform=-1)
        nengo.Connection(ens_degree, ens_d1[1], synapse=0.01, transform=net.z)
        
        # update the WM value stored in the "memory" buffer according to the error passed to D2
        # and the stable WM value stored in the "old" buffer
        nengo.Connection(ens_d2, ens_memory, synapse=0.01, transform=3)
        nengo.Connection(ens_memory, ens_memory, synapse=0.1)
        nengo.Connection(ens_memory, ens_d2, synapse=0.01, transform=-1)
        nengo.Connection(ens_old, ens_d2, synapse=0.01)

        # update the value in "old" to the value in "memory" 
        nengo.Connection(ens_memory, ens_d3, synapse=0.01)
        nengo.Connection(ens_d3, ens_old, synapse=0.01, transform=3)
        nengo.Connection(ens_old, ens_d3, synapse=0.01, transform=-1)
        nengo.Connection(ens_old, ens_old, synapse=0.1)

        # alternatively updates "memory" and "old":
        # the former is updated while the simulus is present,
        # and the latter is updated during the inter-stimulus interval
        nengo.Connection(ens_stim, ens_stop2, synapse=0.01, function=func_stop2)
        nengo.Connection(ens_stim, ens_stop3, synapse=0.01, function=func_stop3)
        nengo.Connection(ens_stop2, ens_d2.neurons, synapse=0.01, transform=w_stop)
        nengo.Connection(ens_stop3, ens_d3.neurons, synapse=0.01, transform=w_stop)

        # probes to decode neural activity into represented quantities
        net.probe_stim = nengo.Probe(node_stim, synapse=None)
        net.probe_stim = nengo.Probe(ens_stim, synapse=0.01)
        net.probe_stop2 = nengo.Probe(ens_stop2, synapse=0.01)
        net.probe_stop3 = nengo.Probe(ens_stop3, synapse=0.01)
        net.probe_weight = nengo.Probe(ens_weight, synapse=0.03)
        net.probe_scaled_weight = nengo.Probe(ens_scaled_weight, synapse=0.03)
        net.probe_d1 = nengo.Probe(ens_d1, synapse=0.03)
        net.probe_d2 = nengo.Probe(ens_d2, synapse=0.01)
        net.probe_d3 = nengo.Probe(ens_d3, synapse=0.01)
        net.probe_memory = nengo.Probe(ens_memory, synapse=0.01)
        net.probe_old = nengo.Probe(ens_old, synapse=0.01)

    return net

def simulate_WM(env, z=0, k=1, seed_sim=0, seed_net=0, progress_bar=True):
    net = build_network_WM(env, seed_net=seed_net, z=z, k=k)
    sim = nengo.Simulator(net, seed=seed_sim, progress_bar=progress_bar)
    with sim:
        sim.run(env.Tall, progress_bar=progress_bar)
    return net, sim

def run_WM(sid, z, k, save=True):
    empirical = pd.read_pickle(f"data/human.pkl").query("sid==@sid")
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
            for o in range(len(observations)):
                n_observations += 1
                tidx = int((n_observations*env.T)/env.dt)-2
                estimate = sim.data[net.probe_memory][tidx][0]
                df = pd.DataFrame([['NEF_WM', sid, trial, stage, estimate]], columns=columns)
                dfs.append(df)
        data = pd.concat(dfs, ignore_index=True)
        if save:
            data.to_pickle(f"data/NEF_WM_{sid}_estimates.pkl")
    return data