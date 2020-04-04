import numpy as np
import pandas as pd
import scipy.stats as stats
import scipy.optimize as opt

from matplotlib import pyplot as plt

from finance.utilities import utils


def annualized_return(start_price, end_price, n_days):
    annual_return = (1+((end_price-start_price)/start_price))**(365/n_days)-1
    return annual_return


def kelly_criterion(predicted_win, predicted_loss, p_win):
    bet_size = (predicted_win * p_win - predicted_loss * (1 - p_win)) / predicted_win
    return bet_size


def get_estimated_loss(df, 
                       profit: str='profit',
                       window_size: int=100,
                       loss: str='median'
                      ):
    df['is_profitable'] = False
    df.loc[df[profit] > 0, 'is_profitable'] = True

    df['profit_rate'] = (df['is_profitable'].rolling(window_size, min_periods=1).sum()
                          /df['is_profitable'].rolling(window_size, min_periods=1).count())
    
    if loss == 'mean':
        df['estimated_loss'] = abs(df[profit]).rolling(window_size, min_periods=1).mean()
    elif loss == 'median':
        df['estimated_loss'] = abs(df[profit]).rolling(window_size, min_periods=1).median()

    return df.drop(['is_profitable'], axis=1)


def greedy_kelly(df,
                 level: str='market_datetime',
                 kelly: str='kelly',
                 price: str='open',
                 profit: str='profit',
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
        positions['trade_profit'] = positions['n_shares'] * positions[profit]
        budget += sum(positions['trade_profit'])
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
    profit: str='profit',
    profit_pred: str='profit_pred',
    trade_profit: str='trade_profit',
    trade: str='trade',
    position: str='position',
    window_size: int=100,
    budget: int=1000
):

    # Performance by stock     
    args = {
        trade: 'count',
        profit: ['sum', 'mean'],
        profit_pred: ['sum', 'mean'],
        trade_profit: 'sum'
    }
    stocks = df.groupby(symbol).agg(args)
    
    # Performance by day
    args = {
        symbol: 'nunique',
        profit: ['sum', 'mean'],
        profit_pred: ['sum', 'mean'],
        trade: 'count',
        position: 'sum',
        trade_profit: 'sum'
    }
    daily = df.groupby(time).agg(args)
    trades = df.groupby(trade).agg(args)
    
    daily['mad'] = (
        daily[trade_profit, 'sum'] 
        - daily[trade_profit, 'sum'].rolling(window_size, min_periods=1).mean()
    ).abs().rolling(window_size, min_periods=1).mean()
    
    plots = {
        'Number of Trades': daily[trade, 'count'],
        'Trading Profits': daily[trade_profit, 'sum'].cumsum(),
        'Profit MAD': daily['mad'].tail(len(daily) - window_size),
        'Max Daily Change': daily[trade_profit, 'sum'].abs().cummax(),
    }
    for plot in plots:
        plt.title(plot)
        plt.plot(plots.get(plot))
        plt.show()
    
    hists = {
        'Daily Trades': daily[trade]['count'],
        'Daily % Change': (
            (budget+daily[trade_profit, 'sum'].cumsum())
            /(budget+daily[trade_profit, 'sum'].cumsum().shift(1))).dropna(),
        'Trades By Stocks': stocks[trade, 'count'],
        'Profits By Stocks': stocks[trade_profit, 'sum'],
    }
    for hist in hists:
        plt.title(hist)
        plt.hist(hists.get(hist))
        plt.show()
    
    bars = {
        'Trades By Direction': trades[trade, 'count'],
        'Profits By Direction': trades[trade_profit, 'sum'],
        'Position Size By Direction': trades[position, 'sum'],
    }
    for bar in bars:
        data = bars.get(bar)
        plt.title(bar)
        plt.bar(data.index, data.values)
        plt.show()
    
    return daily, stocks, trades


def sp_comparison(
    daily=None,
    date_start='2000-01-01',
    date_end='2010-01-01',
    window_size=100
):

    query = f"""
        with raw as (
            select 
                market_datetime::date
                , 100 * (1 - open/lag(open) over(order by market_datetime)) as sp_return
            from yahoo.sp_index
            where market_datetime between '{date_start}'::date - interval '1 day' and '{date_end}')
        select *
        from raw
        where market_datetime >= '{date_start}'
        """
    sp = utils.query_db(query=query)
    
    df = pd.DataFrame(100 * (daily['trade_profit', 'sum'] / daily['position', 'sum']), columns=['portfolio_return'])
    df['day_date'] = df.index.date
    df = sp.merge(df[['day_date', 'portfolio_return']], how='inner', left_on='market_datetime', right_on='day_date')
    
    plt.title('SP & Portfolio Returns')
    plt.plot(df['market_datetime'], df['sp_return'], color='r', label='SP')
    plt.plot(df['market_datetime'], df['portfolio_return'], color='g', label='Portfolio')
    plt.hlines(0, xmin=df['market_datetime'].min(), xmax=df['market_datetime'].max())
    plt.legend()
    plt.show()
    
    plt.title('Portfolio Over SP Returns')
    plt.plot(df['market_datetime'], df['portfolio_return'] - df['sp_return'])
    plt.hlines(0, xmin=df['market_datetime'].min(), xmax=df['market_datetime'].max())
    plt.show()
    
    plt.title('Max Daily Drawdown')
    plt.plot(df['sp_return'].cummin(), label='SP')
    plt.plot(df['portfolio_return'].cummin(), label='Portfolio')
    plt.legend()
    plt.show()

    for col in ['sp_return', 'portfolio_return']:
        df[col + '_mad'] = (
            df[col]
            - df[col].rolling(window_size, min_periods=1).mean()
        ).abs().rolling(window_size, min_periods=1).mean()
    
    plt.title('Portfolio v SP MAD')
    plt.plot(df['sp_return_mad'].tail(len(df) - window_size), label='SP')
    plt.plot(df['portfolio_return_mad'].tail(len(df) - window_size), label='Portfolio')
    plt.legend()
    plt.show()
    
    return df
