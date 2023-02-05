import datetime as dt
from datetime import datetime, timedelta
from sklearn.base import BaseEstimator, DensityMixin
import glob
import pandas as pd
import pytz
import logging
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
import json
import warnings

logger = logging.getLogger(__name__)
