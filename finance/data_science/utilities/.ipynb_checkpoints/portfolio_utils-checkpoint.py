import numpy as np
import pandas as pd
import scipy.stats as stats
import scipy.optimize as opt

from matplotlib import pyplot as plt


def annualized_return(start_price, end_price, n_days):
    annual_return = (1+((end_price-start_price)/start_price))**(365/n_days)-1
    return annual_return


def kelly_criterion(predicted_win, predicted_loss, p_win):
    bet_size = (predicted_win * p_win - predicted_loss * (1 - p_win)) / predicted_win
    return bet_size


def get_estimated_loss(df, 
                       profits: str='profits',
                       window_size: int=100,
                       loss: str='median'
                      ):
    df['is_profitable'] = False
    df.loc[df[profits] > 0, 'is_profitable'] = True

    df['profit_rate'] = (df['is_profitable'].rolling(window_size, min_periods=1).sum()
                          /df['is_profitable'].rolling(window_size, min_periods=1).count())
    
    if loss == 'mean':
        df['estimated_loss'] = abs(df[profits]).rolling(window_size, min_periods=1).mean()
    elif loss == 'median':
        df['estimated_loss'] = abs(df[profits]).rolling(window_size, min_periods=1).median()

    return df.drop(['is_profitable'], axis=1)


def greedy_kelly(df,
                 level: str='market_datetime',
                 kelly: str='kelly',
                 price: str='open',
                 profits: str='profits',
                 budget: float=1000,
                ):
    df['position'] = 0
    df['n_shares'] = 0

    budgets = [budget]
    positions_dfs = []
    for idx, temp in df.groupby(level):
        stocks = temp[temp['kelly'] > 0]
        positions = stocks.loc[stocks.sort_values(by=kelly, ascending=False)[kelly].cumsum() < 1].copy()
        positions['n_shares'] = ((positions[kelly] * budget)/positions[price]).astype(int)
        positions['position'] = positions['n_shares'] * positions[price]
        budget += sum(positions['n_shares'] * positions[profits])
        budgets.append(budget)
        positions_dfs.append(positions)

    return pd.concat(positions_dfs), budgets


def n_largest_kelly(df,
                    n_stocks: int=3,
                    level: str='market_datetime',
                    kelly: str='kelly',
                    price: str='open',
                    profits: str='profits',
                    budget: float=1000,
                   ):
    df['position'] = 0
    df['n_shares'] = 0
    df[kelly + '_adj'] = 0

    positions_dfs = []
    for idx, temp in df.groupby(level):
        stocks = temp[temp[kelly] > 0]
        positions = stocks.nlargest(n_stocks, kelly)
        positions[kelly + '_adj'] = positions[kelly] * (1 / positions[kelly].sum())
        positions['n_shares'] = ((positions[kelly] * budget)/positions[price]).astype(int)
        positions['position'] = positions['n_shares'] * positions[price]
        budget += sum(positions['n_shares'] * positions[profits])
        positions_dfs.append(positions)

    return pd.concat(positions_dfs)


def greedy_kelly_diversified(
    df,
    n_stocks: int=3,
    time: str='market_datetime',
    dimensions: list=[],
    kelly: str='kelly',
    price: str='open',
    profits: str='profits',
    budget: float=1000,
):
    df['position'] = 0
    df['n_shares'] = 0

    positions_dfs = []
    budgets = [budget]
    for idx, day in df.groupby(time):
        allocation = budget / df[dimensions].nunique()
        daily_positions = pd.DataFrame()
        for idx, dimension in day.groupby(dimensions):
            stocks = dimension[dimension[kelly] > 0]
            positions = stocks.loc[stocks.sort_values(by=kelly, ascending=False)[kelly].cumsum() < 1].copy()
            positions['n_shares'] = ((positions[kelly] * allocation.values)/positions[price]).astype(int)
            positions['position'] = positions['n_shares'] * positions[price]
            daily_positions = daily_positions.append(positions)
        budget += sum(daily_positions['n_shares'] * daily_positions[profits])
        budgets.append(budget)
        positions_dfs.append(daily_positions)

    return pd.concat(positions_dfs), budgets


def portfolio_performance(
    df,
    time: str='market_datetime',
    symbol: str='symbol',
    profits: str='profits',
    predicted_profits: str='predicted_profits',
    trade: str='trade',
    position: str='position',
    window_size: int=100,
    budget: int=1000
):
    df['trade_profits'] = df[profits] * df['n_shares']

    # Performance by stock     
    args = {
        trade: 'count',
        profits: ['sum', 'mean'],
        predicted_profits: ['sum', 'mean'],
        'trade_profits': 'sum'
    }
    stocks = df.groupby(symbol).agg(args)
    
    # Performance by day
    args = {
        symbol: 'nunique',
        profits: ['sum', 'mean'],
        predicted_profits: ['sum', 'mean'],
        trade: 'count',
        position: 'sum',
        'trade_profits': 'sum'
    }
    daily = df.groupby(time).agg(args)
    trades = df.groupby(trade).agg(args)
    
    daily['mad'] = (
        daily['trade_profits']['sum'] 
        - daily['trade_profits']['sum'].rolling(window_size, min_periods=1).mean()
    ).abs().rolling(window_size, min_periods=1).mean()
    
    plots = {
        'Number of Trades': daily['trade']['count'],
        'Trading Profits': daily['trade_profits']['sum'].cumsum(),
        'Profit MAD': daily['mad'].tail(len(daily) - window_size),
        'Max Daily Change': daily['trade_profits']['sum'].abs().cummax(),
    }
    for plot in plots:
        plt.title(plot)
        plt.plot(plots.get(plot))
        plt.show()
    
    hists = {
        'Daily Trades': daily['trade']['count'],
        'Daily % Change': (
            (budget+daily['trade_profits']['sum'].cumsum())
            /(budget+daily['trade_profits']['sum'].cumsum().shift(1))).dropna(),
        'Trades By Stocks': stocks['trade']['count'],
        'Profits By Stocks': stocks['trade_profits']['sum'],
    }
    for hist in hists:
        plt.title(hist)
        plt.hist(hists.get(hist))
        plt.show()
    
    bars = {
        'Trades By Direction': trades['trade']['count'],
        'Profits By Direction': trades['trade_profits']['sum'],
        'Position Size By Direction': trades[position]['sum'],
    }
    for bar in bars:
        data = bars.get(bar)
        plt.title(bar)
        plt.bar(data.index, data.values)
        plt.show()
    
    return daily, stocks
