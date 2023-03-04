from scipy.stats import skewnorm, norm
import datetime as dt
from datetime import datetime, timedelta
from sklearn.base import BaseEstimator, DensityMixin
import glob
import pandas as pd
import pytz
import numpy as np
from sklearn.cluster import KMeans
import plotly.graph_objects as go
from sklearn.preprocessing import StandardScaler
from keras.models import Sequential
from keras.layers import Dense, LSTM, Dropout, SpatialDropout1D
from keras.callbacks import EarlyStopping
from plotly.subplots import make_subplots
from scipy.stats import pearsonr
from ta.trend import MACD
from ta.momentum import StochasticOscillator
from scipy.signal import argrelextrema
from sklearn.model_selection import GridSearchCV
from forex_python.converter import CurrencyRates
from numpy.random import rand
from tqdm import tqdm
import json
import warnings

warnings.filterwarnings(action="ignore", message="Mean of empty slice")

TIMEZONE = pytz.timezone("GMT")
INITIAL_TIME = datetime(1970, 1, 1, 00, 00, 00, 000000, tzinfo=TIMEZONE)


class CustomJSONizer(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.bool_):
            return bool(obj)
        elif isinstance(obj, np.int64):
            return int(obj)
        else:
            try:
                return super().default(obj)
            except Exception as e:
                return "Class"


class StockSimulator:
    def __init__(
        self,
        days=7,
        initial_price=0.9545,
        volatility=0.0005,
        drift=0.0000001,
        spread=0.002,
    ) -> None:
        days = 7
        self.initial_price = initial_price

        self.stock = self.create_stock(
            initial_price,
            drift,
            volatility,
            spread,
            days,
        )

    def create_stock(self, initial_price, drift, volatility, spread, days):
        stock = self.create_empty_df(days)
        stock["ask"][0] = initial_price
        random_variates = self.create_random_variates(size=stock.shape[0])
        changes = []
        for i in tqdm(range(1, stock.shape[0])):
            # TODO: Vectorize this implementation
            change = (random_variates[i] * volatility) + drift
            val = stock.iloc[i - 1] + change
            stock.iloc[i] = val
            changes.append(change)
        stock["bid"] = stock["ask"].apply(
            lambda x: x * (1 + (spread * (np.random.rand())))
        )
        self.changes = changes
        return stock

    def create_random_variates(self, size=1000, loc=0, scale=1):
        x = norm.rvs(size=size, scale=scale, loc=loc)
        return x

    def create_empty_df(self, days):
        time = pd.Series(
            pd.date_range(datetime.now(), periods=days * 24 * 60 * 60, freq="S")
        ).values
        values = np.zeros(time.shape).reshape(-1, 1)
        stock = pd.DataFrame(data=values, columns=["ask"], index=time)
        return stock

    def get_stock(self):
        return self.stock

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


def generate_random_symbols(
    symbol_count=1,
    days=7,
    initial_price=0.9545,
    volatility=np.random.rand() * 0.0001,
    drift=0.0,
    trend=0.00000001,
    spread=0.002,
):
    symbols = None
    for symbol in range(symbol_count):
        sm = StockSimulator(
            days=days,
            initial_price=initial_price,
            volatility=volatility,
            drift=drift,
            spread=spread,
        )
        candles = sm.get_candlestick()
        candles["symbol"] = f"SYNTH-{symbol}"
        candles = candles.reset_index()
        candles = candles.rename(columns={"index": "ctmstring"})
        if symbols is None:
            symbols = candles
        else:
            symbols = symbols.append(candles)

    return symbols


def calculate_position(price, vol, contractSize=100000, leverage=5, currency="EUR"):
    position = contractSize * vol * price
    margin = position * (leverage / 100)
    return convert_currency(margin, currency)


def convert_currency(amount, in_currency, out_currency="EUR"):
    c = CurrencyRates()
    return c.convert(in_currency, out_currency, amount) * 1.07


# DATE MANAGING
def get_today():
    return datetime.now().strftime("%m-%d-%Y")


def get_today_ms():
    timestamp = datetime.now()
    t_str = timestamp.strftime("%Y-%m-%d %H:%M:%S.%f")
    t_int = int(timestamp.strftime("%Y%m%d%H%M%S"))
    return t_str, t_int


def xtb_time_to_date(time, local_tz=None):
    initial = INITIAL_TIME
    date = initial + timedelta(milliseconds=time)
    if local_tz:
        local_tz = datetime.now().astimezone().tzinfo
        date_str = (
            date.replace(tzinfo=TIMEZONE).astimezone().strftime("%Y-%m-%d %H:%M:%S.%f")
        )
        return date_str
    else:
        return date.strftime("%Y-%m-%d %H:%M:%S.%f")


def date_to_xtb_time(target):
    target = target.astimezone(TIMEZONE)
    days_diff = (target - INITIAL_TIME).days * 24 * 3600 * 1000
    seconds_diff = (target - INITIAL_TIME).seconds * 1000
    return days_diff + seconds_diff


def get_today_timeString():
    timestamp = datetime.today().astimezone(pytz.timezone("CET"))
    return timestamp.strftime("%a %b %d %H:%M:%S %Z %Y")


def min_max_norm(x, _min, _max):
    return (x - _min) / (_max - _min)


# STOCK DATA RELATED
def adapt_data(df):
    df["close"] = df["open"] + df["close"]
    df["high"] = df["open"] + df["high"]
    df["low"] = df["open"] + df["low"]
    df = df.set_index(pd.to_datetime(df["ctmString"]))

    df = df.dropna()
    df = add_supports(df)

    return df


def cast_candles_to_types(df, digits, dates=True):
    if df is not None:
        df["ctm"] = pd.to_numeric(df["ctm"])
        # Dec 4, 2022, 11:00:00 PM
        if dates:
            df["ctmString"] = pd.to_datetime(
                df["ctmString"], format="%b %d, %Y, %I:%M:%S %p"
            )
            df["ctmString"] = df["ctmString"].dt.strftime("%m/%d/%Y %H:%M:%S")
            df["ctmString"] = pd.to_datetime(df["ctmString"])
        else:
            df["ctmString"] = pd.to_datetime(df["ctmString"])

        df["open"] = pd.to_numeric(df["open"])
        df["close"] = pd.to_numeric(df["close"])
        df["high"] = pd.to_numeric(df["high"])
        df["low"] = pd.to_numeric(df["low"])
        df["vol"] = pd.to_numeric(df["vol"])
        df = df.set_index(df["ctmString"])
        if digits:
            df = numbers_to_decimal(df, digits)
    return df


def cast_ticks_to_types(df):
    if df is not None:
        df["ask"] = pd.to_numeric(df["ask"])
        df["askVolume"] = pd.to_numeric(df["askVolume"], downcast="integer")
        df["bid"] = pd.to_numeric(df["bid"])
        df["bidVolume"] = pd.to_numeric(df["bidVolume"], downcast="integer")
        df["high"] = pd.to_numeric(df["high"])
        df["low"] = pd.to_numeric(df["low"])
        df["level"] = pd.to_numeric(df["level"], downcast="integer")
        df["spreadRaw"] = pd.to_numeric(df["spreadRaw"])
        df["spreadTable"] = pd.to_numeric(df["spreadTable"])
        df["timestamp"] = pd.to_datetime(df["timestamp"])
    return df


def numbers_to_decimal(df, digits):
    for column in ["open", "close", "high", "low", "vol"]:
        df[column] = df[column] / np.power(10, digits)
    return df


# Stock intelligence
def add_supports(df, margin=5):
    df["min"] = df.iloc[argrelextrema(df.close.values, np.less_equal, order=margin)[0]][
        "close"
    ]
    df["max"] = df.iloc[
        argrelextrema(df.close.values, np.greater_equal, order=margin)[0]
    ]["close"]
    return df


def add_rolling_means(df, short, long):
    df["MA_short"] = df["close"].rolling(window=short).mean()
    df["MA_long"] = df["close"].rolling(window=long).mean()
    return df


# Command templates
def baseCommand(commandName, arguments=None):
    if arguments == None:
        arguments = dict()
    return dict([("command", commandName), ("arguments", arguments)])


# Login command for xtb
def loginCommand(userId, password, appName=""):
    return baseCommand("login", dict(userId=userId, password=password, appName=appName))


# Convert return data into pandas dataframe
def return_as_df(returnData):
    if isinstance(returnData, list):
        columns = returnData[0].keys()
        df_dict = {}
        for col in columns:
            df_dict[col] = []
        for r_dict in returnData:
            for col in columns:
                df_dict[col].append(r_dict[col])
        df = pd.DataFrame.from_dict(df_dict)
        return df
    elif isinstance(returnData, dict):
        columns = returnData.keys()
        df_dict = {}
        for col in columns:
            df_dict[col] = []
        for r_dict in returnData:
            for col in columns:
                df_dict[col].append(r_dict[col])
        df = pd.DataFrame.from_dict(df_dict)
        return df
    else:
        return None


def get_line_slope(p1, p2):  # [x1, y1], [x2, y2]
    x1, y1 = p1
    x2, y2 = p2
    return (y2 - y1) / (x2 - x1)


def get_angle_between_lines(m1, m2):
    return np.degrees(np.arctan((m1 - m2) / (1 + (m1 * m2))))


def log_dict(info_dict, logger, no_keys=["Trades", "Crossover", "candles"]):
    str = ""
    for key in info_dict.keys():
        if key not in no_keys:
            new_str = f"\n{key}: {info_dict[key]}"
            str += new_str
    logger.info(str)


# PLOTTING FUNCTIONS
def plot_stock_simple(
    df,
    trades=None,
    crossovers=None,
    symbol="",
    params="",
    profit="",
    show=False,
    min_list=None,
    max_list=None,
    id="ctmstring",
):
    fig = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.01,
        row_heights=[0.9, 0.1],
    )
    fig.add_trace(
        go.Candlestick(
            x=df.index,
            open=df["open"],
            high=df["high"],
            low=df["low"],
            close=df["close"],
            name=symbol,
        ),
        row=1,
        col=1,
    )
    if "short_ma" in list(df.columns):
        fig.add_trace(
            go.Scatter(
                x=df.index,
                y=df["short_ma"],
                opacity=0.7,
                line=dict(color="blue", width=2),
                name="short_ma",
                text=df["angle_short"],
            )
        )
    if "long_ma" in list(df.columns):
        fig.add_trace(
            go.Scatter(
                x=df.index,
                y=df["long_ma"],
                opacity=0.7,
                line=dict(color="orange", width=2),
                name="long_ma",
                text=df["angle_long"],
            )
        )
    if "vol" in list(df.columns):
        fig.add_trace(go.Bar(x=df.index, y=df["vol"], name="Volume"), row=2, col=1)

    if min_list:
        for min_line in min_list:
            fig.add_hline(
                y=min_line,
                name="min",
                line_dash="dash",
                line_color="yellow",
                line_width=3,
                row=1,
                col=1,
            )
    if max_list:
        for max_line in max_list:
            fig.add_hline(
                y=max_line,
                name="max",
                line_dash="dash",
                line_color="blue",
                line_width=3,
                row=1,
                col=1,
            )
    if "trades" in list(df.columns):
        fig.add_trace(
            go.Scatter(
                x=df[df["trades"] == 1].index,
                y=df[df["trades"] == 1]["open"],
                name="Trade Buys",
                mode="markers",
                marker_color="green",
                marker_size=10,
                marker_symbol="cross",
            ),
            row=1,
            col=1,
        )
        fig.add_trace(
            go.Scatter(
                x=df[df["trades"] == -1].index,
                y=df[df["trades"] == -1]["open"],
                name="Trade Short",
                mode="markers",
                marker_color="red",
                marker_size=10,
                marker_symbol="cross",
            ),
            row=1,
            col=1,
        )

    # if crossovers:
    #     for crossover in crossovers:
    #         fig.add_vline(
    #             x=crossover["time"],
    #             name="Cross",
    #             line_dash="dash",
    #             line_color='black',
    #             line_width=3,
    #             row=1,
    #             col=1,
    #         )

    fig.update_layout(
        xaxis_rangeslider_visible=False,
        xaxis2_rangeslider_visible=False,
        title_text=f"{symbol}-{params}-p:{profit}",
    )
    config = {
        "scrollZoom": True,
        "displayModeBar": True,
        "displaylogo": False,
        "toImageButtonOptions": {
            "format": "svg",  # one of png, svg, jpeg, webp
            "filename": "custom_image",
            "height": 1080,
            "width": 1920,
            "scale": 1,  # Multiply title/legend/axis/canvas sizes by this factor
        },
        "modeBarButtonsToAdd": [
            "drawline",
            "drawopenpath",
            "drawclosedpath",
            "drawcircle",
            "drawrect",
            "eraseshape",
        ],
    }

    if show:
        fig.show(config=config)
    else:
        fig.write_html(f"results/{symbol}_config.html", config=config)


def plot_stock_all_trades(
    df,
    trades=None,
    crossovers=None,
    symbol="",
    params="",
    profit="",
    show=False,
    ret=False,
):
    fig = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.01,
        row_heights=[0.9, 0.1],
    )
    fig.add_trace(
        go.Candlestick(
            x=df.index,
            open=df["open"],
            high=df["high"],
            low=df["low"],
            close=df["close"],
            name=symbol,
        ),
        row=1,
        col=1,
    )
    if "short_ma" in list(df.columns):
        fig.add_trace(
            go.Scatter(
                x=df.index,
                y=df["short_ma"],
                opacity=0.7,
                line=dict(color="blue", width=2),
                name="short_ma",
            )
        )
    if "long_ma" in list(df.columns):
        fig.add_trace(
            go.Scatter(
                x=df.index,
                y=df["long_ma"],
                opacity=0.7,
                line=dict(color="orange", width=2),
                name="long_ma",
            )
        )
    if "vol" in list(df.columns):
        fig.add_trace(go.Bar(x=df.index, y=df["vol"], name="Volume"), row=2, col=1)

    for cont, trade in enumerate(trades):
        fig.add_trace(
            go.Scatter(
                x=[trade["open_time"]],
                y=[trade["open_price"]],
                name="Trade Open",
                legendgroup=f"trade_{cont}",
                mode="markers",
                marker_color="green",
                marker_size=10,
                marker_symbol="cross",
            ),
            row=1,
            col=1,
        )
        fig.add_trace(
            go.Scatter(
                x=[trade["close_time"]],
                y=[trade["close_price"]],
                name="Trade Close",
                text=f"Profit: {trade['profit']}",
                legendgroup=f"trade_{cont}",
                mode="markers",
                marker_color="red",
                marker_size=10,
                marker_symbol="cross",
            ),
            row=1,
            col=1,
        )
        if "sl" in list(trade.keys()):
            fig.add_trace(
                go.Scatter(
                    x=[trade["open_time"], trade["close_time"]],
                    y=[trade["sl"]],
                    name="Stop Loss",
                    legendgroup=f"trade_{cont}",
                    mode="lines",
                    line=dict(color="red", width=5),
                ),
                row=1,
                col=1,
            )
        if "tp" in list(trade.keys()):
            fig.add_trace(
                go.Scatter(
                    x=[trade["open_time"], trade["close_time"]],
                    y=[trade["tp"]],
                    name="Take Profit",
                    legendgroup=f"trade_{cont}",
                    mode="lines",
                    line=dict(color="green", width=5),
                ),
                row=1,
                col=1,
            )

    fig.update_layout(
        xaxis_rangeslider_visible=False,
        xaxis2_rangeslider_visible=False,
        title_text=f"{symbol}-{params}-p:{profit}",
    )
    config = {
        "scrollZoom": True,
        "displayModeBar": True,
        "displaylogo": False,
        "toImageButtonOptions": {
            "format": "svg",  # one of png, svg, jpeg, webp
            "filename": "custom_image",
            "height": 1080,
            "width": 1920,
            "scale": 1,  # Multiply title/legend/axis/canvas sizes by this factor
        },
        "modeBarButtonsToAdd": [
            "drawline",
            "drawopenpath",
            "drawclosedpath",
            "drawcircle",
            "drawrect",
            "eraseshape",
        ],
    }

    if show:
        fig.show(config=config)
    else:
        fig.write_html(f"results/{symbol}/{symbol}_config.html", config=config)
    if ret:
        return fig


def plot_stock_trade(
    df,
    trade=None,
    id=0,
    crossovers=None,
    symbol="",
    params="",
    profit="",
    show=True,
    path=None,
):
    fig = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.01,
        row_heights=[0.9, 0.1],
    )
    fig.add_trace(
        go.Candlestick(
            x=df["ctmstring"],
            open=df["open"],
            high=df["high"],
            low=df["low"],
            close=df["close"],
            name=symbol,
        ),
        row=1,
        col=1,
    )
    if "short_ma" in list(df.columns):
        fig.add_trace(
            go.Scatter(
                x=df["ctmstring"],
                y=df["short_ma"],
                opacity=0.7,
                line=dict(color="blue", width=2),
                name="short_ma",
            )
        )
    if "long_ma" in list(df.columns):
        fig.add_trace(
            go.Scatter(
                x=df["ctmstring"],
                y=df["long_ma"],
                opacity=0.7,
                line=dict(color="orange", width=2),
                name="long_ma",
            )
        )
    if "vol" in list(df.columns):
        fig.add_trace(
            go.Bar(x=df["ctmstring"], y=df["vol"], name="Volume"), row=2, col=1
        )

    if trade is not None:
        fig.add_trace(
            go.Scatter(
                x=[trade["open_time"]],
                y=[trade["open_price"]],
                name="Trade Open",
                mode="markers",
                marker_color="black",
                marker_size=10,
                marker_symbol="cross",
            ),
            row=1,
            col=1,
        )
        fig.add_hline(
            y=trade["sl"],
            name="SL",
            line_dash="dash",
            line_color="red",
            line_width=3,
            row=1,
            col=1,
        )
        fig.add_hline(
            y=trade["tp"],
            name="TP",
            line_dash="dash",
            line_color="green",
            line_width=3,
            row=1,
            col=1,
        )
        if "close_time" in trade.keys():
            fig.add_trace(
                go.Scatter(
                    x=[trade["close_time"]],
                    y=[trade["close_price"]],
                    name="Trade Close",
                    mode="markers",
                    marker_color="blue",
                    marker_size=10,
                    marker_symbol="circle",
                ),
                row=1,
                col=1,
            )

    # if crossovers:
    #     for crossover in crossovers:
    #         fig.add_vline(
    #             x=crossover["time"],
    #             name="Cross",
    #             line_dash="dash",
    #             line_color='black',
    #             line_width=3,
    #             row=1,
    #             col=1,
    #         )

    fig.update_layout(
        xaxis_rangeslider_visible=False,
        xaxis2_rangeslider_visible=False,
        title_text=f"{symbol}-{params}-p:{profit}",
    )
    config = {
        "scrollZoom": True,
        "displayModeBar": True,
        "displaylogo": False,
        "toImageButtonOptions": {
            "format": "svg",  # one of png, svg, jpeg, webp
            "filename": "custom_image",
            "height": 1080,
            "width": 1920,
            "scale": 1,  # Multiply title/legend/axis/canvas sizes by this factor
        },
        "modeBarButtonsToAdd": [
            "drawline",
            "drawopenpath",
            "drawclosedpath",
            "drawcircle",
            "drawrect",
            "eraseshape",
        ],
    }

    if show:
        fig.show(config=config)
    else:
        if path:
            fig.write_html(f"results/{path}.html", config=config)
        else:
            fig.write_html(f"results/trades/{symbol}_trade{id}.html", config=config)


def plot_stock(df, params=None, symbol=None, return_fig=False, fig=None):
    if not fig:
        fig = make_subplots(
            rows=4,
            cols=1,
            shared_xaxes=True,
            vertical_spacing=0.01,
            row_heights=[0.5, 0.1, 0.2, 0.2],
        )
    fig.add_trace(
        go.Candlestick(
            x=df["ctmString"],
            open=df["open"],
            high=df["high"],
            low=df["low"],
            close=df["close"],
            name=symbol,
        ),
        row=1,
        col=1,
    )
    if "short_ma" in list(df.columns):
        fig.add_trace(
            go.Scatter(
                x=df.index,
                y=df["short_ma"],
                opacity=0.7,
                line=dict(color="blue", width=2),
                name="short_ma",
            )
        )
    if "long_ma" in list(df.columns):
        fig.add_trace(
            go.Scatter(
                x=df.index,
                y=df["long_ma"],
                opacity=0.7,
                line=dict(color="orange", width=2),
                name="long_ma",
            )
        )
    MACD1, STOCH1 = get_macd_stoch(df)
    fig.add_trace(go.Bar(x=df.index, y=df["vol"]), row=2, col=1)
    fig.add_trace(go.Bar(x=df.index, y=MACD1.macd_diff()), row=3, col=1)
    fig.add_trace(
        go.Scatter(x=df.index, y=MACD1.macd(), line=dict(color="black", width=2)),
        row=3,
        col=1,
    )
    fig.add_trace(
        go.Scatter(x=df.index, y=MACD1.macd_signal(), line=dict(color="blue", width=1)),
        row=3,
        col=1,
    )
    fig.add_trace(
        go.Scatter(x=df.index, y=STOCH1.stoch(), line=dict(color="black", width=2)),
        row=4,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=df.index, y=STOCH1.stoch_signal(), line=dict(color="blue", width=1)
        ),
        row=4,
        col=1,
    )

    if params:
        fig.add_hline(
            y=params["open_price"],
            name="BP",
            line_dash="dash",
            line_color="yellow",
            line_width=3,
            row=1,
            col=1,
        )
        fig.add_hline(
            y=params["sl"],
            name="SL",
            line_dash="dash",
            line_color="red",
            line_width=3,
            row=1,
            col=1,
        )
        fig.add_hline(
            y=params["close_price"],
            name="CP",
            line_dash="dash",
            line_color="green",
            line_width=3,
            row=1,
            col=1,
        )
    fig.update_layout(
        xaxis_rangeslider_visible=False,
        xaxis2_rangeslider_visible=False,
        # xaxis=dict(type="category", categoryorder="category ascending"),
        # xaxis2=dict(type="category", categoryorder="category ascending"),
        # xaxis3=dict(type="category", categoryorder="category ascending"),
        # xaxis4=dict(type="category", categoryorder="category ascending"),
    )
    # fig.update_xaxes(
    #     rangebreaks=[
    #         dict(bounds=[17, 9], pattern="hour"),
    #         dict(bounds=["sat", "mon"]),
    #     ]
    # )
    # fig.update_xaxes(rangebreaks=[dict(bounds=["sat", "mon"])])
    fig.show()
    if return_fig:
        return fig


def plot_v_line(fig, day, color):
    fig.add_vline(
        x=day,
        name="BP",
        line_dash="dash",
        line_color=color,
        line_width=3,
        row=1,
        col=1,
    )
    return fig


def get_macd_stoch(df):
    macd = MACD(close=df["close"], window_slow=26, window_fast=12, window_sign=9)
    STOCH = StochasticOscillator(
        high=df["high"], close=df["close"], low=df["low"], window=14, smooth_window=3
    )
    return macd, STOCH


def trading_loss(y, y_pred, **kwargs):
    y_true = np.zeros_like(y)
    return np.sum(y_pred - y_true)


# def create_pdf(sd, mean, alfa):
#     #invertire il segno di alfa
#     x = skewnorm.rvs(alfa, size=1000000)
#     def calc(k, sd, mean):
#       return (k*sd)+mean
#     x = calc(x, sd, mean) #standard distribution
#     return x

# def create_empty_df(days, start=datetime.today()):
#     #creare un empty DataFrame con le date
#     empty = pd.Series(
#         pd.date_range(start=start, periods=days, freq="D")
#     )
#     empty = pd.DataFrame(empty)
#     #si tagliano ore, minuti, secondi
#     empty

#     #si tagliano ore, minuti, secondi
#     empty.index = [str(x)[0:empty.shape[0]] for x in list(empty.pop(0))]
#     empty

#     #final dataset con values
#     stock = pd.DataFrame([x for x in range(0, empty.shape[0])])
#     stock.index = empty.index
#     return stock

# def sinmulate_stock(initial_price, drift, volatility, trend, days):


class TradingEstimator(BaseEstimator, DensityMixin):
    def __init__(
        self,
        period,
        capital,
        symbol,
        short_enabled,
        leverage,
        contractSize,
        lotStep,
        short_ma=None,
        long_ma=None,
        min_angle=None,
        out=None,
        show=False,
        fig=False,
    ) -> None:
        super().__init__()
        self.candles = None
        self.period = period
        self.capital = capital
        self.symbol = symbol
        self.short_ma = short_ma
        self.long_ma = long_ma
        self.min_angle = min_angle
        self.out = out
        self.short_enabled = short_enabled
        self.leverage = leverage
        self.contractSize = contractSize
        self.lotStep = lotStep
        self.show = show
        if not fig:
            self.fig = fig
        else:
            self.fig = make_subplots(
                rows=4,
                cols=1,
                shared_xaxes=True,
                vertical_spacing=0.01,
                row_heights=[0.5, 0.1, 0.2, 0.2],
            )

    def fit(self, X, y=None, **kwargs):
        self.candles = X
        return self

    def score(self, X, y=None):
        return self.back_test(X)

    def test(self, X, parameters, test_period="1D", show=False):
        self.short_ma = parameters["short_ma"]
        self.long_ma = parameters["long_ma"]
        self.min_angle = parameters["min_angle"]
        self.out = parameters["out"]
        return self.back_test(X, test_period=test_period, show=show)

    def back_test(self, df=None, test_period=None, show=False):
        if df is None:
            df = self.candles
        period = self.period
        capital = self.capital
        symbol = self.symbol
        min_angle = self.min_angle
        out = self.out
        short_ma = self.short_ma
        long_ma = self.long_ma
        trend = 0
        short_enabled = self.short_enabled
        show = show

        df = add_rolling_means(df, short=short_ma, long=long_ma)
        if test_period is not None:
            df = df.iloc[int(df.shape[0] * period) :, :]
        else:
            df = df.iloc[: int(df.shape[0] * period), :]
        if show:
            plot_stock(df, symbol=symbol)

        start_date = df["ctmString"][0].strftime("%Y-%m-%d")
        is_bought = False
        is_short = False
        in_up_crossover = False
        in_down_crossover = False
        profits = []
        profit = 0
        potential_profits = [0]
        if show:
            print("XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX")
            print("")
            print("")
            print(f"Start test from {start_date}")
            print(f"    Short selling: {short_enabled}")
            print(f"    Capital at risk: {capital}")
            print(f"    Trend: {trend}")

        for step in range(df.shape[0]):
            prev_tick = df.iloc[step - 1, :]
            tick = df.iloc[step, :]
            tick_day = str(tick["ctmString"])

            if is_bought:
                diff = tick["low"] - prev_tick["low"]
                profit = tick["low"] - buy_price
                if show:
                    print(f"Tick: {start_date}")
                    print(f"Sell Price {tick['low']}")
                    print(f"Buy price {buy_price}")
                    print(f"Profit {profit}")
                    print(f"Diff {diff}")
                    print(f"Potential profits {potential_profits}")

                if profit < -(buy_price * out):
                    # If the profit has gone down by a 5%, we get out
                    (
                        is_bought,
                        is_short,
                        profits,
                        potential_profits,
                    ) = self.sell_position(profits, profit)

                elif profit > 0:
                    potential_profits.append(diff)

                    if (prev_tick["low"] - buy_price) > (tick["low"] - buy_price) * (
                        1 + out
                    ):
                        # If the current movement has decreased over 5% from the previous one, we get out
                        (
                            is_bought,
                            is_short,
                            profits,
                            potential_profits,
                        ) = self.sell_position(profits, profit)
                        if show:
                            print("Sold because of downward trend")
                else:
                    potential_profits.append(diff)

            if is_short:
                diff = prev_tick["low"] - tick["low"]
                profit = buy_price - tick["low"]
                if show:
                    print(f"Tick: {start_date}")
                    print(f"Sell Price {tick['low']}")
                    print(f"Buy price {buy_price}")
                    print(f"Profit {profit}")
                    print(f"Diff {diff}")
                    print(f"Potential profits {potential_profits}")

                if profit < -(buy_price * out):
                    # If the profit has gone down by a 5%, we get out
                    (
                        is_bought,
                        is_short,
                        profits,
                        potential_profits,
                    ) = self.sell_position(profits, profit)

                elif profit > 0:
                    potential_profits.append(diff)

                    if (prev_tick["low"] - buy_price) > (tick["low"] - buy_price) * (
                        1 + out
                    ):
                        # If the current movement has decreased over 5% from the previous one, we get out
                        (
                            is_bought,
                            is_short,
                            profits,
                            potential_profits,
                        ) = self.sell_position(profits, profit)
                        if show:
                            print("Sold because of downward trend")
                else:
                    potential_profits.append(diff)

            if (
                (
                    (prev_tick["MA_short"] < prev_tick["MA_long"])
                    and (tick["MA_short"] > tick["MA_long"])
                    or (prev_tick["MA_short"] > prev_tick["MA_long"])
                    and (tick["MA_short"] < tick["MA_long"])
                )
                or (in_up_crossover and not is_bought)
                or (in_down_crossover and not is_bought)
            ):
                if show:
                    print(f"CROSSOVER at {tick_day}")

                balance = np.sum(profits) + capital
                c_price = tick["high"]

                ma_short_slope = get_line_slope(
                    [
                        (step - 1) / df.shape[0],
                        prev_tick["MA_short"] / df["MA_short"][: step - 1].max(),
                    ],
                    [
                        step / df.shape[0],
                        tick["MA_short"] / df["MA_short"][:step].max(),
                    ],
                )
                ma_long_slope = get_line_slope(
                    [
                        (step - 1) / df.shape[0],
                        prev_tick["MA_long"] / df["MA_long"][: step - 1].max(),
                    ],
                    [step / df.shape[0], tick["MA_long"] / df["MA_long"][:step].max()],
                )
                crossover_angle = get_angle_between_lines(ma_short_slope, ma_long_slope)

                if show:
                    print(f"    Balance: {balance}")
                    print(f"    Price: {c_price}")
                    print(f"    Crossover angle: {crossover_angle}")

            if (
                (prev_tick["MA_short"] < prev_tick["MA_long"])
                and (tick["MA_short"] > tick["MA_long"])
            ) or (in_up_crossover and not is_bought):
                if show:
                    print(f"Short term upward crossover")
                if is_bought:
                    (
                        is_bought,
                        is_short,
                        profits,
                        potential_profits,
                    ) = self.sell_position(profits, profit)
                if np.abs(crossover_angle) > min_angle:
                    if np.sum(profits) + capital > tick["high"]:
                        (is_bought, is_short, buy_price) = self.buy_position(tick)
                    else:
                        if show:
                            print(f"Not enough money to buy")
                else:
                    in_up_crossover = True
                    if show:
                        print(f"Crossover angle to small {crossover_angle}")

            if (
                (prev_tick["MA_short"] > prev_tick["MA_long"])
                and (tick["MA_short"] < tick["MA_long"])
                or (in_down_crossover and not is_short)
            ):
                if show:
                    print(f"Short term downward crossover")
                if is_bought:
                    (
                        is_bought,
                        is_short,
                        profits,
                        potential_profits,
                    ) = self.sell_position(profits, profit)
                if short_enabled:
                    if np.abs(crossover_angle) > min_angle:
                        if np.sum(profits) + capital > tick["high"]:
                            (is_bought, is_short, buy_price) = self.buy_position(
                                tick, short_enabled
                            )
                        else:
                            if show:
                                print(f"Not enough money to buy")
                    else:
                        in_down_crossover = True
                        if show:
                            print(f"Crossover angle to small {crossover_angle}")

            if step == df.shape[0] - 1 and (is_bought or is_short):
                (is_bought, is_short, profits, potential_profits) = self.sell_position(
                    profits, profit
                )

        # FINISHED BACKTEST
        if len(profits) > 0:
            np_profits = np.array(profits)
            lossing_trades = np_profits[np_profits < 0]
            winning_trades = np_profits[np_profits > 0]
            if show:
                print(f"Profits: {np.nansum(profits)}")
                print(
                    f"Winning/Lossing trades: {len(winning_trades)}|{len(lossing_trades)}"
                )
                print(
                    f"Average transaction profit: w:{np.nanmean(winning_trades)} | l: {np.nanmean(lossing_trades)}"
                )
                print("")
                print("")
                print("End test")
                print("XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX")
        else:
            profits = 0
        return np.sum(profits)

    def sell_position(self, profits, profit):
        is_bought = False
        is_short = False
        profits.append(profit)
        potential_profits = [0]
        return is_bought, is_short, profits, potential_profits

    def buy_position(self, tick, is_short=False):
        is_bought = True
        buy_price = tick["high"]
        cost = (
            tick["high"] * self.contractSize / ((1 / self.leverage) * 100)
        ) * self.lotStep
        return is_bought, is_short, buy_price

    def look_for_suitable_symbols_v1(self, df, symbol_type, capital, max_risk):
        # TODO look for suitable symbols
        # Look for symbols with:
        #   - Volatility bigger than spread [DONE LATER]
        #   - Trader_type products
        #   - Ask comparable to our max_risk
        try:
            df = df[df["categoryName"] == symbol_type]
            df["min_price"] = (
                df["ask"] * df["lotMin"] * df["contractSize"] * (df["leverage"] / 100)
            )
            df = df[df["min_price"] <= (capital * max_risk)]
            df["spread_percentage"] = (df["ask"] - df["bid"]) / df["ask"]
            df = df.sort_values(by=["spread_percentage"])
            return df
        except Exception as e:
            raise LogicError
