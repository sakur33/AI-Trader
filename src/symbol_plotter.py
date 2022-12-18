import pickle as pkl
from utils import *
import warnings
import os

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


sym1_s = "TGNA.US"
sym2_s = "IOVA.US"
sym1 = glob.glob(f"{data_path}*{sym1_s}*.pickle")[0]
df_sym1 = adapt_data(pd.read_pickle(sym1))
sym2 = glob.glob(f"{data_path}*{sym2_s}*.pickle")[0]
df_sym2 = adapt_data(pd.read_pickle(sym2))

plot_stock(df_sym1, df_sym2, symbols=[sym1_s, sym2_s])
