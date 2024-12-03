import numpy as np
import scipy
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import time
import statsmodels.formula.api as smf
import statsmodels.api as sm
from fit import likelihood, stat_fit_scipy
palette = sns.color_palette('tab10')
sns.set_palette(palette)
sns.set(context='paper', style='white', font="cmr10", font_scale=1.2)
plt.rcParams['axes.formatter.use_mathtext'] = True

def measure_lag(data):
    sids = data['sid'].unique()
    model_types = data['type'].unique()
    columns = ['type', 'sid', 'trial', 'neighbors', 'stage', 'lag', 'action equals lagged observation']
    dfs = []
    for model_type in model_types:
        for sid in sids:
            print('type', model_type, 'sid', sid)
            trials = data.query('sid==@sid')['trial'].unique()
            for trial in trials:
                subdata = data.query("trial==@trial and sid==@sid")
                stages = subdata['stage'].unique()
                neighbors = len(subdata['who'].unique()) - 1
                for stage in stages:
                    observations = subdata.query("stage<=@stage")['color'].to_numpy()
                    action = subdata.query("stage==@stage")['action'].unique()[0]
                    # print('stage', stage, 'action', action, 'obs', observations)
                    for o in range(len(observations)):
                        lag = o+1
                        lagged_obs = observations[-lag]
                        action_eq_lagged_obs = 1 if action==lagged_obs else 0
                        # print(lag, lagged_obs, action_eq_lagged_obs)
                        df = pd.DataFrame([[
                            model_type,
                            sid,
                            trial,
                            neighbors,
                            stage,
                            lag,
                            action_eq_lagged_obs
                        ]], columns=columns)
                        dfs.append(df)
        lag_data = pd.concat(dfs, ignore_index=True)
        lag_data.to_pickle(f"data/lagged_reruns_{model_type}.pkl")

# human = pd.read_pickle(f"data/human.pkl")
# human['type'] = 'human'
# nan = np.nan
# nonnef = pd.read_pickle("data/all_scipy2_reruns.pkl").query("type not in ['DGrd', 'DGrds', 'DGrdp', 'DGrdpz', 'NEF-WM', 'NEF-RL']")
# nefwm = pd.read_pickle("data/NEF-WM_loadzk_reruns.pkl")
# nefrl = pd.read_pickle("data/NEF-RL_loadzk_reruns.pkl")
# reruns = pd.concat([human, nonnef, nefwm, nefrl], ignore_index=True)
# reruns = reruns.query("type != @nan")
# reruns.to_pickle(f"data/all_reruns.pkl")
# print(reruns)
reruns = pd.read_pickle(f"data/all_reruns.pkl")
measure_lag(reruns)