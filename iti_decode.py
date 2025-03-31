import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import nengo
import scipy
import time
import sys
from fit import *
from NEF_WM import *
from NEF_RL import *

if __name__ == '__main__':
	model_type = sys.argv[1]
	sid = int(sys.argv[2])
	empirical = pd.read_pickle(f"data/carrabin.pkl").query("sid==@sid")
	trials = empirical['trial'].unique()
	params = pd.read_pickle(f"data/{model_type}_carrabin_mar30_params.pkl").query("sid==@sid")
	alpha = params['alpha'].to_numpy()[0]
	lambd = params['lambda'].to_numpy()[0]
	n_neurons = params['n_neurons'].to_numpy()[0]
	start = time.time()
	if model_type=='NEF_syn':
		data = run_NEF_syn("carrabin", sid=sid, alpha=alpha, z=0, lambd=lambd, n_neurons=n_neurons, pretrain=True, iti=True)
	if model_type=='NEF_rec':
		data = run_NEF_rec("carrabin", sid=sid, alpha=alpha, z=0, lambd=lambd, n_neurons=n_neurons, pretrain=True, iti=True)
	data.to_pickle(f"data/{model_type}_{sid}_iti_decode.pkl")
	end = time.time()
	print(f"runtime {(end-start)/60:.4}")