from scipy.stats import skewnorm, norm
import pandas as pd
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt


class StockSimulator:
    def __init__(
        self,
        initial_price=np.random.rand(),
        drift=np.random.rand(),
        volatility=np.random.rand(),
        trend=np.random.rand(),
        spread=np.random.rand(),
        days=[7],
    ) -> None:
        self.initial_price = initial_price
        self.drift = drift
        self.volatilitie = volatility
        self.trend = trend
        self.spread = spread
        self.days = days

        drift = drift / (days * 3600)
        volatility = volatility / (days * 3600)

        # self.create_random_variates(trend, volatility, drift)

        self.stock = self.create_stock(
            initial_price,
            drift,
            volatility,
            trend,
            spread,
            days,
        )

    def get_stock(self):
        return self.stock

    def create_stock(self, initial_price, drift, volatility, trend, spread, days):
        stock = self.create_empty_df(days)
        stock["ask"][0] = initial_price
        changes = []
        for i in range(1, stock.shape[0]):
            change = (skewnorm.rvs(trend) * volatility) + drift
            val = stock.iloc[i - 1] * (1 + change)
            stock.iloc[i] = val
            changes.append(change)
        stock["bid"] = stock["ask"].apply(
            lambda x: x * (1 + (spread * (np.random.rand())))
        )
        self.random_variates = changes
        return stock

    def create_random_variates(self, trend, volatility, drift):
        x = skewnorm.rvs(trend, size=10000)
        x = (x * volatility) + drift
        self.random_variates = x

    def create_empty_df(self, days):
        time = pd.Series(
            pd.date_range(datetime.now(), periods=days * 24 * 3600, freq="S")
        ).values
        values = np.zeros(time.shape).reshape(-1, 1)
        stock = pd.DataFrame(data=values, columns=["ask"], index=time)
        return stock

    def get_candlestick(self, frame="1min", price="ask"):
        candles = (
            self.stock[price]
            .resample(frame)
            .agg(
                {
                    "open": "first",
                    "high": "max",
                    "low": "min",
                    "close": "last",
                }
            )
        )
        return candles
