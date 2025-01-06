import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import nengo
import pandas as pd
import sys
import optuna
import mysql.connector
import scipy
from WM2 import *
from RL2 import *

def objective(trial, model_type, sid):
	if model_type=='NEF_RL':
		z = trial.suggest_float("z", 0.0, 2.0, step=0.01)
		b = trial.suggest_float("b", 0.0, 0.4, step=0.001)
		inv_temp = trial.suggest_float("inv_temp", 0.0, 10.0, step=0.01)
		s = [1,b,b,b]
		data = run_RL(sid, z, s, save=False)
		# from likelihood function in fit.py
		NLL = 0
		human = pd.read_pickle(f"data/human.pkl").query("sid==@sid")
		trials = human['trial'].unique()
		stages = human['stage'].unique()
		for trial in trials:
			for stage in stages:
				subdata = data.query("sid==@sid & trial==@trial & stage==@stage")
				expectation = subdata['estimate'].to_numpy()[0]
				act = human.query("trial==@trial and stage==@stage")['action'].unique()[0]
				prob = scipy.special.expit(inv_temp*expectation)
				NLL -= np.log(prob) if act==1 else np.log(1-prob)
	return NLL

if __name__ == '__main__':

	model_type = sys.argv[1]
	sid = int(sys.argv[2])
	study_name = f"{model_type}_{sid}"
	optuna_trials = 0

	# objective(None, model_type, sid)
	# raise

	host = "olsn-18b4-db.c.dartmouth.edu"
	user = "psipeter"
	password = ""

	study = optuna.create_study(
		study_name=study_name,
		storage=f"mysql+mysqlconnector://{user}:{password}@{host}/{study_name}",
		load_if_exists=True,
		direction="minimize")
	study.optimize(lambda trial: objective(trial, model_type, sid), n_trials=optuna_trials)

	best_params = study.best_trial.params
	param_names = ["type"]
	saved_params = [model_type]
	for key, value in best_params.items():
		param_names.append(key)
		saved_params.append(value)
	print(f"{len(study.trials)} trials completed. Best parameters:")
	print(param_names)
	print(saved_params)
	fitted_params = pd.DataFrame([saved_params], columns=param_names)
	fitted_params.to_pickle(f"data/{model_type}_{sid}_params.pkl")