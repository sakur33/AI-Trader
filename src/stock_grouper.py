import glob
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

today = get_today()
show = False
picks = glob.glob("../data/*_1440.pickle")

correlations = np.zeros((len(picks), len(picks)))
names = []
for cont1, pick1 in enumerate(picks):
    df1 = adapt_data(pd.read_pickle(pick1))
    symbol1 = pick1.split("_")[0].split("\\")[-1]
    names.append(symbol1)
    corrs = []
    for cont2, pick2 in enumerate(picks):
        df2 = adapt_data(pd.read_pickle(pick2))
        symbol2 = pick2.split("_")[0].split("\\")[-1]
        df1_len = len(df1["open"].values)
        df2_len = len(df2["open"].values)
        if df1_len > df2_len:
            r, p = pearsonr(df1["open"].values[-df2_len:], df2["open"].values)
        else:
            r, p = pearsonr(df1["open"].values, df2["open"].values[-df1_len:])
        corrs.append(r)
    correlations[cont1, :] = np.array(corrs)

df = pd.DataFrame(data=correlations, columns=names, index=names)
if show:
    show_heatmap(df, df.columns.to_list(), df.columns.to_list())

group_n = 3
model = KMeans(n_clusters=group_n)
model.fit(df.values)
groups = model.predict(df.values)

symbol_dict = {}
for cont, symbol in enumerate(names):
    symbol_dict[symbol] = groups[cont]

counts = list(np.zeros(group_n, dtype=int))
clusters = []
for i in range(group_n):
    clusters.append([])

for key in symbol_dict.keys():
    counts[symbol_dict[key]] += 1
    clusters[symbol_dict[key]].append(key)

group_dict = {
    "Date": today,
    "Clusters_n": group_n,
    "Clusters": clusters,
    "Occurrences": counts,
    "Groups": symbol_dict,
}

with open("../clusters/grouper_" + today + ".pickle", "wb") as f:
    pkl.dump(group_dict, f, pkl.HIGHEST_PROTOCOL)

if show:
    show_clusters(group_n, symbol_dict, counts)
