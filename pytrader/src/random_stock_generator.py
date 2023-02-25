from scipy.stats import skewnorm
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def simulate_stock(initial_price, drift, volatility, trend, days):

  def create_pdf(sd, mean, alfa):
    #invertire il segno di alfa
    x = skewnorm.rvs(alfa, size=1000000) 
    def calc(k, sd, mean):
      return (k*sd)+mean
    x = calc(x, sd, mean) #standard distribution

    #graph pdf
    #pd.DataFrame(x).hist(bins=100)

    #pick one random number from the distribution
    #formally I would use cdf, but I just have to pick randomly from the 1000000 samples
    #np.random.choice(x)
    return x

  def create_empty_df(days):
    #creare un empty DataFrame con le date
    empty = pd.Series(
        pd.date_range("2019-01-01", periods=days, freq="D")
    )
    empty = pd.DataFrame(empty)
    #si tagliano ore, minuti, secondi
    empty

    #si tagliano ore, minuti, secondi
    empty.index = [str(x)[0:empty.shape[0]] for x in list(empty.pop(0))]
    empty

    #final dataset con values
    stock = pd.DataFrame([x for x in range(0, empty.shape[0])])
    stock.index = empty.index
    return stock

  #skeleton
  stock = create_empty_df(days)

  #initial price
  stock[0][0] = initial_price

  #create entire stock DataFrame
  x = create_pdf(volatility, drift, trend)
  for _ in range(1, stock.shape[0]):
    stock.iloc[_] = stock.iloc[_-1]*(1+np.random.choice(x))

  return stock
