import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import nengo
import pandas as pd
import sys
from WM import *

model_type = sys.argv[1]
sid = int(sys.argv[2])

if model_type=='WM':
	z = float(sys.argv[3])
	k = float(sys.argv[4])
	data = run_WM(sid, z, k)