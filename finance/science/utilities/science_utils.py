"""portfolio utils"""
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
    lines: tuple,
    title: str = 'Groups Plot',
    n_plots: int = 10,
    error_plot: bool = False,
):
    """Generate multiple plots (or error plots) by groups"""
    n = 0
    plt.plot()
    for label, group in df.groupby(groups):
        plt.title(title + ' ' + label)
        if not error_plot:
            for line in lines:
                plt.plot(group[line], label=line)

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
