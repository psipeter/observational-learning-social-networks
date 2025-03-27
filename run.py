import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import nengo
import pandas as pd
import sys
import time
from NEF_WM import *
from NEF_RL import *
from NEF_syn import *
from NEF_rec import *
from fit import *

dataset = sys.argv[1]
model_type = sys.argv[2]
sid = int(sys.argv[3])
paramfile = sys.argv[4]
params = pd.read_pickle(f"data/{paramfile}_{dataset}_{sid}_params.pkl")
start = time.time()

if dataset=='carrabin':
	if model_type=='NEF_WM':
		alpha = params['alpha'].unique()[0]
		n_all = params['n_all'].unique()[0]
		lambd = params['lambda'].unique()[0]
		data = run_WM("carrabin", sid, alpha=alpha, z=0, lambd=lambd, n_memory=n_all, n_neurons=n_all, n_error=n_all)
		param_list = [model_type, sid, alpha, n_all, lambd]
		param_names = ['type', 'sid', 'alpha', 'n_all', 'lambda']
	if model_type=='NEF_RL':
		alpha = params['alpha'].unique()[0]
		n_all = params['n_all'].unique()[0]
		lambd = params['lambda'].unique()[0]
		data = run_RL("carrabin", sid, alpha=alpha, z=0, lambd=lambd, n_neurons=n_all, n_learning=n_all, n_error=n_all)
		param_list = [model_type, sid, alpha, n_all, lambd]
		param_names = ['type', 'sid', 'alpha', 'n_all', 'lambda']
	if model_type=='NEF_syn':
		alpha = params['alpha'].unique()[0]
		n_neurons = params['n_neurons'].unique()[0]
		lambd = params['lambda'].unique()[0]
		data = run_NEF_syn("carrabin", sid, alpha=alpha, z=0, lambd=lambd, n_neurons=n_neurons)
		param_list = [model_type, sid, alpha, n_neurons, lambd]
		param_names = ['type', 'sid', 'alpha', 'n_neurons', 'lambda']
	if model_type=='NEF_rec':
		alpha = params['alpha'].unique()[0]
		n_neurons = params['n_neurons'].unique()[0]
		lambd = params['lambda'].unique()[0]
		data = run_NEF_rec("carrabin", sid, alpha=alpha, z=0, lambd=lambd, n_neurons=n_neurons)
		param_list = [model_type, sid, alpha, n_neurons, lambd]
		param_names = ['type', 'sid', 'alpha', 'n_neurons', 'lambda']
	loss = QID_loss([], model_type, sid)
	error = mean_loss([], model_type, sid)
	performance_data = pd.DataFrame([[model_type, sid, loss, error]], columns=['type', 'sid', 'loss', 'error'])
	performance_data.to_pickle(f"data/{model_type}_{dataset}_{sid}_performance.pkl")
	fitted_params = pd.DataFrame([param_list], columns=param_names)
	fitted_params.to_pickle(f"data/{model_type}_{dataset}_{sid}_params.pkl")

if dataset=='jiang':
	if model_type=='NEF_WM':
		alpha = params['alpha'].unique()[0]
		lambd = params['lambda'].unique()[0]
		n_all = params['n_all'].unique()[0]
		z = params['z'].unique()[0]
		beta = params['beta'].unique()[0]
		data = run_WM("jiang", sid, alpha=alpha, z=z, lambd=lambd, n_memory=n_all, n_neurons=n_all, n_error=n_all)
		param_list = [model_type, sid, alpha, z, n_all, lambd, beta]
		param_names = ['type', 'sid', 'alpha', 'z', 'n_all', 'lambda', 'beta']
	if model_type=='NEF_RL':
		alpha = params['alpha'].unique()[0]
		lambd = params['lambda'].unique()[0]
		n_all = params['n_all'].unique()[0]
		z = params['z'].unique()[0]
		beta = params['beta'].unique()[0]
		data = run_RL("jiang", sid, alpha=alpha, z=z, lambd=lambd, n_neurons=n_all, n_learning=n_all, n_error=n_all)
		param_list = [model_type, sid, alpha, z, n_all, lambd, beta]
		param_names = ['type', 'sid', 'alpha', 'z', 'n_all', 'lambda', 'beta']
	if model_type=='NEF_syn':
		alpha = params['alpha'].unique()[0]
		lambd = params['lambda'].unique()[0]
		neurons = params['neurons'].unique()[0]
		z = params['z'].unique()[0]
		beta = params['beta'].unique()[0]
		data = run_NEF_syn("jiang", sid, alpha=alpha, z=z, lambd=lambd, n_neurons=neurons)
		param_list = [model_type, sid, alpha, z, neurons, lambd, beta]
		param_names = ['type', 'sid', 'alpha', 'z', 'neurons', 'lambda', 'beta']
	if model_type=='NEF_rec':
		alpha = params['alpha'].unique()[0]
		lambd = params['lambda'].unique()[0]
		neurons = params['neurons'].unique()[0]
		z = params['z'].unique()[0]
		beta = params['beta'].unique()[0]
		data = run_NEF_rec("jiang", sid, alpha=alpha, z=z, lambd=lambd, n_neurons=neurons)
		param_list = [model_type, sid, alpha, z, neurons, lambd, beta]
		param_names = ['type', 'sid', 'alpha', 'z', 'neurons', 'lambda', 'beta']
	NLL = NLL_loss([beta], model_type, sid)
	columns = ['type', 'sid', 'loss']
	performance_data = pd.DataFrame([[model_type, sid, NLL]],columns=columns)
	performance_data.to_pickle(f"data/{model_type}_{dataset}_{sid}_performance.pkl")
	fitted_params = pd.DataFrame([param_list], columns=param_names)
	fitted_params.to_pickle(f"data/{model_type}_{dataset}_{sid}_params.pkl")

end = time.time()
print(f"runtime {(end-start)/60:.4} min")