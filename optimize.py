import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import nengo
import pandas as pd
import sys
import optuna
import mysql.connector
from WM import *

def objective(trial, model_type, sid):
	if model_type=='WM':
		z = trial.suggest_float("z", 0.5, 2.0, step=0.01)
		k = trial.suggest_float("k", 0.5, 2.0, step=0.01)
		data = run_WM(sid, z, k, save=False)
		errors = data.query("type=='model-WM'")['error'].to_numpy()
		loss = np.sum(errors) / len(errors)
		return loss

if __name__ == '__main__':

	model_type = sys.argv[1]
	sid = int(sys.argv[2])
	study_name = f"{model_type}_{sid}"
	optuna_trials = 2

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