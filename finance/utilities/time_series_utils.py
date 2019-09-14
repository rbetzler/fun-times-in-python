import math
import pandas as pd
import statsmodels.tsa.stattools as sm_tools
import statsmodels.tsa.arima_model as stats_arima
from sklearn.metrics import mean_squared_error, mean_absolute_error
import matplotlib.pyplot as plt

from finance.utilities import utils


class ARIMA:
    def __init__(self, series=pd.DataFrame, datetimes=None, forecast_date=None,
                 alpha=.05, limit=1000, frequency='B', p=1, d=1, q=1):
        self.series = series
        self.datetimes = datetimes
        self.forecast_date = pd.to_datetime(forecast_date)
        self.alpha = alpha
        self.limit = limit
        self.frequency = frequency
        self.p = p
        self.d = d
        self.q = q

    @property
    def model_df(self):
        df = pd.DataFrame(data=self.series)
        df.columns = ['vals']

        df.index = pd.DatetimeIndex(self.datetimes.dt.date)
        df.index.names = ['date']
        df.sort_index()

        df = df.resample(self.frequency).bfill()
        df.index.freq = self.frequency
        return df

    @staticmethod
    def model_forecast_helper(df=None, forecast_date=None, p=0, d=0, q=0, frequency='B') -> tuple:
        training_set = df[df.index < forecast_date]['vals']
        dates = df[df.index < forecast_date].index
        params = p, d, q
        model = stats_arima.ARIMA(training_set, params, dates=dates, freq=frequency).fit()
        forecast = model.forecast()[0][0]
        aic = model.aic
        bic = model.bic
        return forecast, aic, bic

    @property
    def model_forecast(self):
        forecast = self.model_forecast_helper(df=self.model_df,
                                              forecast_date=self.forecast_date,
                                              p=self.p,
                                              d=self.d,
                                              q=self.q,
                                              frequency=self.frequency)
        return forecast

    @property
    def optimal_params(self, criterion='mse') -> pd.DataFrame:
        performances = []
        df = self.model_df
        frequency = self.frequency
        for forecast_date in utils.iter_date_range(self.forecast_date, df.index.max()):
            for p in range(0, self.p):
                for d in range(0, self.d):
                    for q in range(0, self.q):
                        forecast, aic, bic = self.model_forecast_helper(df=df, forecast_date=forecast_date,
                                                                        p=p, d=d, q=q, frequency=frequency)
                        actual = df[df.index == forecast_date]
                        performance = {'forecast_date': forecast_date,
                                       'p': p,
                                       'd': d,
                                       'q': q,
                                       'aic': aic,
                                       'bic': bic,
                                       'mse': mean_squared_error(y_true=actual, y_pred=[forecast]),
                                       'mae': mean_absolute_error(y_true=actual, y_pred=[forecast])}
                        performances.append(performance)
        perf_df = pd.DataFrame(performances).groupby(['p', 'd', 'q']).mean()
        optimal_param = perf_df[perf_df[criterion] == perf_df[criterion].min()]
        return optimal_param

    @property
    def acf_vals(self):
        acf = sm_tools.acf(self.series.head(self.limit), alpha=self.alpha, fft=False)
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
        plt.title('PACF of ' + self.series.name)
        plt.xlabel('Lags')
        plt.plot(self.pacf_vals[0], linestyle='-', color='b')
        plt.plot([0, 40], [(1.96 / math.sqrt(1000)), (1.96 / math.sqrt(1000))], linestyle='-', color='r')
        plt.show()
        pass


if __name__ == '__main__':
    query = """
        select
            e.symbol
            , e.market_datetime
            , e.open
            , e.high
            , e.low
            , e.close
            , e.volume
            , f.high_52
            , f.low_52
            , f.dividend_amount
            , f.pe_ratio
            , f.quick_ratio
            , f.current_ratio
        from td.equities as e
        left join td.fundamentals as f
            on f.symbol = e.symbol
        where e.symbol = 'BA'
        order by e.market_datetime
        """
    df = utils.query_db(query=query)
    var = ARIMA(series=df['open'],
                datetimes=df['market_datetime'],
                p=3,
                forecast_date='2019-09-02').optimal_params
    print(var)
