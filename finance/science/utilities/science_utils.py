"""portfolio utils"""
import datetime
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def encode_one_hot(df: pd.DataFrame, columns: str or list):
    """Add one hot endcoding columns to pandas dataframe"""
    if isinstance(columns, str):
        columns = [columns]

    for column in columns:
        one_hot_encoding = pd.get_dummies(df[column], prefix=column)
        df = df.join(one_hot_encoding)

    return df


def plot_groups(
    df: pd.DataFrame,
    groups: list,
    lines: str or list,
    n_plots: int = 10,
    error_plot: bool = False,
    title: str = 'Groups Plot',
    xaxis_name: str = 'market_datetime',
    xaxis_ticks: int = 5,
):
    """Generate multiple plots (or error plots) by groups"""
    n = 0
    lines = [lines] if isinstance(lines, str) else lines

    plt.plot()
    for label, group in df.groupby(groups):
        plt.title(title + ' ' + label)
        if not error_plot:
            for line in lines:
                plt.plot(group[xaxis_name], group[line], label=line)
                if isinstance(group[xaxis_name].values[0], datetime.date):
                    ticks = pd.to_datetime(group['market_datetime'])
                else:
                    ticks = group[xaxis_name]
                plt.xticks([ticks.quantile(x) for x in np.linspace(0, 1, xaxis_ticks)])

        else:
            plt.plot(
                group[lines[0]] - group[lines[1]],
                label='Error ' + label,
            )
            plt.hlines(0, xmin=group.index.min(), xmax=group.index.max())

        plt.legend()
        plt.show()
        n += 1
        if n > n_plots:
            break


def annualized_return(start_price, end_price, n_days):
    annual_return = (1 + ((end_price - start_price) / start_price)) ** (365 / n_days) - 1
    return annual_return


def kelly_criterion(predicted_win, predicted_loss, p_win):
    bet_size = (predicted_win * p_win - predicted_loss * (1 - p_win)) / predicted_win
    return bet_size
