import glob
import os
import pandas as pd
import pickle as pkl
import numpy as np
from datetime import datetime
from utils import adapt_data, get_today, show_clusters, show_heatmap
from scipy.stats import pearsonr
import plotly.express as px
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings
import copy as cp
from utils import *

warnings.filterwarnings("ignore", category=FutureWarning)

today = get_today()
curr_path = os.path.dirname(os.path.realpath(__file__))
data_path = curr_path + "../../data/"
symbol_path = curr_path + "../../symbols/"
cluster_path = curr_path + "../../clusters/"
model_path = curr_path + "../../model/"
docs_path = curr_path + "../../docs/"

with open(f"{cluster_path}grouper_" + today + ".pickle", "rb") as f:
    group_dict = pkl.load(f)
picks = glob.glob(f"{data_path}*.pickle")

for group in group_dict["Clusters"]:
    print("--------------------------------------------------------")
    print(f"GROUP: {group}")

    for pick1 in picks:
        if any(symbol1 in pick1 for symbol1 in group):
            symbol1 = pick1.split("_")[0].split("\\")[-1]
            new_group = cp.deepcopy(group)
            new_group.remove(symbol1)
            for pick2 in picks:
                if any(symbol2 in pick2 for symbol2 in new_group):
                    symbol2 = pick2.split("_")[0].split("\\")[-1]
                    try:
                        df1 = adapt_data(pd.read_pickle(pick1))
                        df2 = adapt_data(pd.read_pickle(pick2))
                    except Exception as e:
                        print(f"Exception loading 1 '{pick1}': {e}")
                        print(f"Exception loading 2 '{pick2}': {e}")

                    print(f"pick1: {symbol1} | pick2: {symbol2}")
                    m_shift, r_max, r_org = find_maximum_correlation_shift(df1, df2)
                    if m_shift != 0:
                        print(
                            f"Shift stats: max: {m_shift} | max corr: {r_max} | orig: {r_org}"
                        )
