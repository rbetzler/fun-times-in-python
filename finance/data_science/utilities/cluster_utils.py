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


def encode_one_hot(df, column):
    for val in df[column].unique():
        df[val] = 0
        df.loc[df[column] == val, val] = 1
    return df


def normalize(df, column, subset=None):
    if subset:
        for val in df[subset].unique():
            df.loc[df[subset] == val, column] = (df.loc[df[subset] == val, column] -
                                                 df.loc[df[subset] == val, column].min()) \
                                                / (df.loc[df[subset] == val, column].max() -
                                                   df.loc[df[subset] == val, column].min())
    else:
        df[column] = (df[column] - df[column].min()) / (df[column].max() - df[column].min())
    return df


if __name__ == '__main__':
    query = """
        select distinct
            market_datetime
            , open
        from td.equities
        where symbol = 'BA'
        and market_datetime between '2017-01-01' and '2019-01-01'
        order by market_datetime
        """
    df = utils.query_db(query=query)
    x = KMeans(series=df['open']).plot_cluster_hist()
    print(x)
