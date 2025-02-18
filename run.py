import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import nengo
import pandas as pd
import sys
import time
from NEF_WM import *
from NEF_RL import *
from fit import *

dataset = sys.argv[1]
model_type = sys.argv[2]
sid = int(sys.argv[3])
paramfile = sys.argv[4]
params = pd.read_pickle(f"data/{paramfile}_{dataset}_{sid}_params.pkl") if paramfile!="none" else []
start = time.time()

if dataset=='carrabin':
	if model_type=='NEF_WM':
		alpha = params['alpha'].unique()[0]
		n_memory = params['n_memory'].unique()[0]
		n_error = params['n_error'].unique()[0]
		data = run_WM("carrabin", sid, alpha=alpha, z=0, n_learning=n_learning, n_error=n_error)
		param_list = [model_type, sid, alpha, n_learning, n_error]
		param_names = ['type', 'sid', 'alpha', 'n_learning', 'n_error']
		loss = qid_abs_loss([], model_type, sid)
	if model_type=='NEF_RL':
		alpha = params['alpha'].unique()[0]
		n_learning = params['n_learning'].unique()[0]
		n_error = params['n_error'].unique()[0]
		data = run_RL("carrabin", sid, alpha=alpha, z=0, n_learning=n_learning, n_error=n_error)
		param_list = [model_type, sid, alpha, n_learning, n_error]
		param_names = ['type', 'sid', 'alpha', 'n_learning', 'n_error']
		loss = qid_abs_loss([], model_type, sid)
	performance_data = pd.DataFrame([[model_type, sid, loss]], columns=['type', 'sid', 'loss'])
	performance_data.to_pickle(f"data/{model_type}_{dataset}_{sid}_performance.pkl")
	fitted_params = pd.DataFrame([param_list], columns=param_names)
	fitted_params.to_pickle(f"data/{model_type}_{dataset}_{sid}_params.pkl")

if dataset=='jiang':
	if model_type=='NEF_WM':
		z = params['z'].unique()[0]
		inv_temp = params['inv_temp'].unique()[0]
		data = run_WM(dataset, sid, z)
		param_list = [model_type, sid, z, inv_temp]
	if model_type=='NEF_RL':
		z = params['z'].unique()[0]
		b = params['b'].unique()[0]
		inv_temp = params['inv_temp'].unique()[0]
		data = run_RL(dataset, sid, z, s=[3,b,b,b])
		# activities = activity_RL(sid, z, s=[3,b,b,b])
		param_list = [model_type, sid, z, b, inv_temp]
	NLL = likelihood(param_list, model_type, sid)
	mcfadden_r2 = compute_mcfadden(NLL, sid)
	columns = ['type', 'sid', 'NLL', 'McFadden R2']
	performance_data = pd.DataFrame([[model_type, sid, NLL, mcfadden_r2]],columns=columns)
	performance_data.to_pickle(f"data/{model_type}_{dataset}_{sid}_performance.pkl")
	param_names = get_param_names(model_type)
	fitted_params = pd.DataFrame([param_list], columns=param_names)
	fitted_params.to_pickle(f"data/{model_type}_{dataset}_{sid}_params.pkl")

end = time.time()
print(f"runtime {(end-start)/60:.4} min")