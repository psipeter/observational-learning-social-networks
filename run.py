import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import nengo
import pandas as pd
import sys
from WM2 import *
from RL2 import *
from fit import *

model_type = sys.argv[1]
sid = int(sys.argv[2])
paramfile = sys.argv[3]
params = pd.read_pickle(f"data/{paramfile}_{sid}_params.pkl")

if model_type=='NEF_WM':
	z = params['z'].unique()[0] if sys.argv[4]=='load' else float(sys.argv[4])
	k = params['k'].unique()[0] if sys.argv[5]=='load' else float(sys.argv[5])
	inv_temp = params['inv_temp'].unique()[0] if sys.argv[6]=='load' else float(sys.argv[6])
	data = run_WM(sid, z, k)
	param_list = [model_type, z, k, inv_temp]

if model_type=='NEF_RL':
	z = params['z'].unique()[0] if sys.argv[4]=='load' else float(sys.argv[4])
	b = params['b'].unique()[0] if sys.argv[5]=='load' else float(sys.argv[5])
	inv_temp = params['inv_temp'].unique()[0] if sys.argv[6]=='load' else float(sys.argv[6])
	data = run_RL(sid, z, s=[3,b,b,b])
	param_list = [model_type, z, b, inv_temp]

NLL = likelihood(param_list, model_type, sid)
mcfadden_r2 = compute_mcfadden(NLL, sid)
columns = ['type', 'sid', 'NLL', 'McFadden R2']
performance_data = pd.DataFrame([[model_type, sid, NLL, mcfadden_r2]],columns=columns)
performance_data.to_pickle(f"data/{model_type}_{sid}_performance.pkl")
param_names = get_param_names(model_type)
fitted_params = pd.DataFrame([param_list], columns=param_names)
fitted_params.to_pickle(f"data/{model_type}_{sid}_params.pkl")