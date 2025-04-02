import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import nengo
import scipy
import pandas as pd
from uniform_encoders import *

class EnvironmentCount():
	def __init__(self, dataset, sid, trial, lambd=0, T=1, dt=0.001, dim_context=10, seed_env=0, iti_noise=0):
		self.T = T
		self.dt = dt
		self.sid = sid
		self.lambd = lambd
		self.iti_noise = iti_noise
		self.trial = trial
		self.dataset = dataset
		self.empirical = pd.read_pickle(f"data/{dataset}.pkl").query("sid==@sid & trial==@trial")
		self.dim_context = dim_context
		self.rng = np.random.RandomState(seed=seed_env)
		self.sampler = ScatteredHypersphere(surface=True)
		# self.context = self.rng.rand(self.dim_context)  # TODO - needs to be unitary
		# self.context = self.context / np.linalg.norm(self.context)
		self.context_vectors = self.sampler.sample(self.dim_context, self.dim_context, rng=self.rng)
		self.context_color = self.context_vectors[0]
		self.context_iti = self.context_vectors[int(self.dim_context/2)]
		if self.dataset=='carrabin':
			self.Tall = 5*self.T - self.dt
			self.stages = range(1, 6)
		if self.dataset=='jiang':
			self.n_neighbors = len(self.empirical['who'].unique()) - 1
			self.Tall = 3*self.T + 3*self.n_neighbors*self.T - self.dt
			self.stages = range(4)
		if self.dataset=='yoo':
			self.Tall = 30*self.T - self.dt
			self.stages = range(1, 31)
		# create input arrays
		self.colors = []
		self.weights = []
		self.contexts = []
		self.neighbor_degrees = []
		self.noise = []
		self.obs_times = []
		self.iti_times = []
		tt = int(self.T / self.dt / 2)
		zeros = np.zeros((tt, 1)).flatten()
		ones = np.ones((tt, 1))
		long_zeros = np.zeros((5*tt, 1)).flatten()
		long_ones = np.ones((5*tt, 1))
		if self.dataset=='carrabin':
			for stage in self.stages:
				color = self.empirical.query("stage==@stage")['color'].unique()[0]
				weight = 1 / (stage+2)**self.lambd
				iti_noise = self.rng.normal(0, self.iti_noise, size=ones.shape[0])
				neighbor_degree = 0
				self.colors.extend(color * ones)
				self.colors.extend(0.000 * ones)
				self.weights.extend(weight * ones)
				self.weights.extend(0.000 * ones)
				self.noise.extend(zeros)
				self.noise.extend(iti_noise)
				self.contexts.extend(self.context_color * ones)
				self.contexts.extend(self.context_iti * ones)
				self.neighbor_degrees.extend(neighbor_degree * ones)
				self.neighbor_degrees.extend(neighbor_degree * ones)
				self.obs_times.append(stage*tt*2-tt)
				self.iti_times.append(stage*tt*2)
		if self.dataset=='jiang':
			for stage in self.stages:
				if stage==0:
					color = self.empirical.query("stage==@stage")['color'].to_numpy()[0]
					weight = 1 / (stage+1)**self.lambd
					neighbor_degree = 0
					iti_noise = self.rng.normal(0, self.iti_noise, size=ones.shape[0])
					self.colors.extend(color * long_ones)
					self.colors.extend(0.000 * ones)
					self.weights.extend(weight * long_ones)
					self.weights.extend(0.000 * ones)
					self.noise.extend(long_zeros)
					self.noise.extend(iti_noise)
					self.contexts.extend(self.context_color * long_ones)
					self.contexts.extend(self.context_iti * ones)
					self.neighbor_degrees.extend(neighbor_degree * long_ones)
					self.neighbor_degrees.extend(neighbor_degree * ones)
					self.obs_times.append(tt*6 - tt)
					self.iti_times.append(tt*6)
				else:
					for n in range(self.n_neighbors):
						color = self.empirical.query("stage==@stage")['color'].to_numpy()[n]
						weight = 1 / ((stage-1)*self.n_neighbors+n+2)**self.lambd
						neighbor_degree = 0 if stage==1 else self.empirical.query("stage==@stage")['RD'].to_numpy()[n]
						iti_noise = self.rng.normal(0, self.iti_noise, size=ones.shape[0])
						self.colors.extend(color * ones)
						self.colors.extend(0.000 * ones)
						self.weights.extend(weight * ones)
						self.weights.extend(0.000 * ones)
						self.noise.extend(zeros)
						self.noise.extend(iti_noise)
						self.contexts.extend(self.context_color * ones)
						self.contexts.extend(self.context_iti * ones)
						self.neighbor_degrees.extend(neighbor_degree * ones)
						self.neighbor_degrees.extend(neighbor_degree * ones)
					self.obs_times.append(tt*6 + stage*self.n_neighbors*tt*2 - tt)
					self.iti_times.append(tt*6 + stage*self.n_neighbors*tt*2)
		if self.dataset=='yoo':
			for stage in self.stages:
				obs = self.empirical.query("stage==@stage")['observation'].unique()[0]
				weight = 1 / stage**self.lambd
				iti_noise = self.rng.normal(0, self.iti_noise, size=ones.shape[0])
				neighbor_degree = 0
				self.colors.extend(obs * ones)  # too lazy to rename
				self.colors.extend(0.000 * ones)
				self.weights.extend(weight * ones)
				self.weights.extend(0.000 * ones)
				self.noise.extend(zeros)
				self.noise.extend(iti_noise)
				self.contexts.extend(self.context_color * ones)
				self.contexts.extend(self.context_iti * ones)
				self.neighbor_degrees.extend(neighbor_degree * ones)
				self.neighbor_degrees.extend(neighbor_degree * ones)
				self.obs_times.append(stage*tt*2-tt)
				self.iti_times.append(stage*tt*2)
		self.colors = np.array(self.colors)
		self.weights = np.array(self.weights)
		self.contexts = np.array(self.contexts)
		self.neighbor_degrees = np.array(self.neighbor_degrees)
		self.noise = np.array(self.noise)
	def sample_color(self, t):
		tidx = int(t/self.dt)
		return self.colors[tidx]
	def sample_weight(self, t):
		tidx = int(t/self.dt)
		return self.weights[tidx]
	def sample_context(self, t):
		tidx = int(t/self.dt)
		return self.contexts[tidx]
	def sample_neighbor_degree(self, t):
		tidx = int(t/self.dt)
		return self.neighbor_degrees[tidx]
	def sample_noise(self, t):
		tidx = int(t/self.dt)
		return self.noise[tidx]