"""portfolio utils"""
import datetime
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def encode_one_hot(
    df: pd.DataFrame,
    column: str,
    keys: list,
) -> pd.DataFrame:
    """Add one hot encoding columns to pandas dataframe"""
    df_keys = pd.DataFrame(0, index=np.arange(len(df)), columns=keys)
    data = df.join(df_keys, rsuffix=f'_{column}')
    for key in keys:
        data.loc[data[column] == key, key] = 1
    return data


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
        plt.title(title + ' ' + str(label))
        if not error_plot:
            for line in lines:
                plt.plot(group[xaxis_name], group[line], label=line)
                if isinstance(group[xaxis_name].values[0], datetime.date):
                    ticks = pd.to_datetime(group[xaxis_name])
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
    """
    Kelly Criterion for bet sizing.

    Bet Size = P(Win) - P(Loss) / Net Winnings

    Where, Net Winnings = Win / Loss
    """
    bet_size = p_win - ((1 - p_win) / predicted_win / predicted_loss)
    return bet_size
