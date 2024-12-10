import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import nengo
import pandas as pd
import sys
# from WM import *
from WM2 import *
from RL import *

model_type = sys.argv[1]
sid = int(sys.argv[2])

if model_type=='WM':
	z = float(sys.argv[3])
	data = run_WM(sid, z)
	# k = float(sys.argv[4])
	# data = run_WM(sid, z, k)

if model_type=='RL':
	z = float(sys.argv[3])
	k = float(sys.argv[4])
	data = run_RL(sid, z, k)
	# learning_rate = float(sys.argv[5])
	# data = run_RL(sid, z, k, learning_rate)