import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import cluster as skcluster

from finance.utilities import utils


class KMeans:
    def __init__(self, series=pd.DataFrame, n_clusters=10, random_state=0):
        self.series = series
        self.n_clusters = n_clusters
        self.random_state = random_state

    @property
    def cluster(self):
        clusters = skcluster.KMeans(n_clusters=self.n_clusters,
                                    random_state=self.random_state).fit(self.series.values.reshape(-1, 1))
        return clusters

    @property
    def predicted_clusters(self):
        return self.cluster.predict(self.series.values.reshape(-1, 1))

    @property
    def predicted_cluster_centers(self):
        return self.cluster.cluster_centers_

    @property
    def predicted_values(self):
        return self.predicted_cluster_centers[self.predicted_clusters]

    def plot_fit(self, length=1000):
        plt.figure()
        plt.title('KMeans Fit')
        plt.plot(self.series.head(length).index.values, self.predicted_values[:length], color='r', linestyle='-.')
        plt.plot(self.series.head(length))
        plt.show()

    def plot_cluster_hist(self, time_weight=False):
        plt.figure()
        plt.title('KMeans Predictions')
        if time_weight:
            plt.hist(self.predicted_values, weights=np.linspace(start=1,
                                                                stop=self.predicted_values.shape[0],
                                                                num=self.predicted_values.shape[0]))
        else:
            plt.hist(self.predicted_values)
        plt.show()


def encode_one_hot(df, columns, drop_columns=False):
    for column in columns:
        one_hot_encoding = pd.get_dummies(df[column], prefix=column)
        df = df.join(one_hot_encoding)
    return df


def normalize(df, column, subset=None, window_size=10):
    if subset:
        for val in df[subset].unique():
            df.loc[df[subset] == val, column] = (df.loc[df[subset] == val, column] -
                                                 df.loc[df[subset] == val, column].rolling(window_size).min()) \
                                                / (df.loc[df[subset] == val, column].rolling(window_size).max() -
                                                   df.loc[df[subset] == val, column].rolling(window_size).min())
    else:
        df[column] = (df[column] - df[column].min()) / (df[column].max() - df[column].min())
    return df


def plot_groupby(df,
                 groupby,
                 lines=['prediction', 'actuals'],
                 title='Groupby Plot', 
                 n_plots=10,
                 n_ticks=10,
                 error_plot=False):
    n = 0
    plt.plot()
    for label, group in df.groupby(groupby):
        plt.title(title + ' ' + label)
        if not error_plot:
            for line in lines:
                plt.plot(group[line], 
                         label=line)
        else:
            plt.plot(group[lines[0]]-group[lines[1]], 
                     label='Error ' + label)
            plt.hlines(0, xmin=group.index.min(), xmax=group.index.max())
        plt.legend()
        plt.show()
        n += 1
        if n > n_plots:
            break