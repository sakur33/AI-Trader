from trader_utils import StockSimulator
import pickle as pkl
import os

if os.path.exists("stock.pkl"):
    with open("stock.pkl", "rb") as f:
        sm = pkl.load(f)
    print("Loaded")
else:
    sm = StockSimulator(
        days=7,
        initial_price=0.9545,
        volatility=0.0005, # 43,2 daily volatility
        drift=0.0000001, # 0,00864 daily drift
        trend=0.001, # 86,4 daily trend
        spread=0.002, # 172,8 daily spread
    )
    stock = sm.get_stock()
    candles = sm.get_candlestick()

    with open("stock.pkl", "wb") as f:
        pkl.dump(sm, f, pkl.HIGHEST_PROTOCOL)
    print("Created")

# Measure Volatility
stock = sm.get_stock()
o_mean = stock["ask"].mean()
o_std = stock["ask"].std()
o_min = stock["ask"].min()
o_max = stock["ask"].max()
h_means = stock["ask"].resample("D").mean()
h_stds = stock["ask"].resample("D").std()

print(f"Increase: {stock['ask'][-1] - stock['ask'][0]}")
print(f"    Mean value: {o_mean}")
print(f"    Max return: {o_max - o_mean}")
print(f"    Max drawdown: {o_min - o_mean}")
print(f"    Volatility: {o_std}")
print(f"Daily values: {h_means.values}")
print(f"Daily volatility: {h_stds.values}")
