import math
import pandas as pd
import statsmodels.tsa.stattools as sm_tools

import matplotlib.pyplot as plt


class ARIMA:
    def __init__(self,
                 series=pd.DataFrame,
                 alpha=.05,
                 limit=1000):
        self.series = series
        self.alpha = alpha
        self.limit = limit

    @property
    def acf_vals(self):
        acf = sm_tools.acf(self.series.head(self.limit), alpha=self.alpha)
        return acf

    def acf_plot(self):
        plt.figure()
        plt.title('ACF of ' + self.series.name)
        plt.xlabel('Lags')
        plt.plot(self.acf_vals[0], linestyle='-', color='b')
        plt.plot(self.acf_vals[1][:, 0], linestyle='--', color='grey')
        plt.plot(self.acf_vals[1][:, 1], linestyle='--', color='grey')
        plt.plot([0, 40], [(1.96 / math.sqrt(1000)), (1.96 / math.sqrt(1000))], linestyle='-', color='r')
        plt.show()
        pass

    @property
    def pacf_vals(self):
        pacf = sm_tools.pacf(self.series.head(self.limit), alpha=self.alpha)
        return pacf

    def pacf_plot(self):
        plt.figure()
        plt.figure()
        plt.title('PACF of ' + self.series.name)
        plt.xlabel('Lags')
        plt.plot(self.pacf_vals[0], linestyle='-', color='b')
        plt.plot([0, 40], [(1.96 / math.sqrt(1000)), (1.96 / math.sqrt(1000))], linestyle='-', color='r')
        plt.show()
        pass


if __name__ == '__main__':
    x = pd.DataFrame([1,2,3,4,5])
    var = ARIMA(series=x).pacf_plot()
    print(var)
