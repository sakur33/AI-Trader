import glob
import pandas as pd
import pickle as pkl
import numpy as np
from datetime import datetime
from utils import adapt_data, get_today
from scipy.stats import pearsonr
import plotly.express as px
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import plotly.graph_objects as go
from plotly.subplots import make_subplots

today = get_today()
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
print(df.head(9))
fig = px.imshow(df)

group_n = 3
model = KMeans(n_clusters=group_n)
model.fit(df.values)
groups = model.predict(df.values)

symbol_dict = {}
for cont, symbol in enumerate(names):
    symbol_dict[symbol] = groups[cont]

counts = list(np.zeros(group_n, dtype=int))
for key in symbol_dict.keys():
    counts[symbol_dict[key]] += 1

group_dict = {
    "Date": today,
    "Clusters": group_n,
    "Occurrences": counts,
    "Groups": symbol_dict,
}

with open("../models/grouper" + today, "wb") as f:
    pkl.dump(group_dict, f, pkl.HIGHEST_PROTOCOL)

with open("../models/grouper" + today, "rb") as f:
    group_dict = pkl.load(f)

for i in range(group_n):
    grid_side = int(np.ceil(np.sqrt(counts[i])))
    fig = make_subplots(rows=grid_side + 1, cols=grid_side + 1)
    cont_col = 0
    cont_row = 0
    for key in symbol_dict.keys():
        if symbol_dict[key] == i:
            cont_col += 1
            pick = glob.glob(f"../data/{key}*_1440.pickle")[0]
            print(pick)
            df = adapt_data(pd.read_pickle(pick))
            print(f"{grid_side} | {cont}: {cont % grid_side + 1}")
            print(f"{grid_side} | {cont}: {cont // grid_side + 1}")
            fig.add_trace(
                go.Scatter(
                    x=df.index,
                    y=df["close"],
                    mode="lines",
                    name=key
                ),
                col=cont_col % grid_side + 1,
                row=cont_row // grid_side + 1,
            )
            cont_row += 1

    fig.show()
    input()
