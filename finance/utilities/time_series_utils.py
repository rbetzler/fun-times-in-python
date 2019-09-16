import math
import pandas as pd
import statsmodels.tsa.stattools as sm_tools
import statsmodels.tsa.arima_model as stats_arima
from sklearn.metrics import mean_squared_error, mean_absolute_error
import matplotlib.pyplot as plt

from finance.utilities import utils


class ARIMA:
    def __init__(self, series=pd.DataFrame, datetimes=None, forecast_date=None,
                 alpha=.05, limit=100, frequency='B', p=1, d=1, q=1, criterion='mse'):
        self.series = series
        self.datetimes = datetimes
        self.forecast_date = pd.to_datetime(forecast_date)
        self.alpha = alpha
        self.limit = limit
        self.frequency = frequency
        self.p = p
        self.d = d
        self.q = q
        self.criterion = criterion

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

    @property
    def differences(self):
        df = self.model_df
        df['diff_one'] = df - df.shift(1)
        df['diff_two'] = df['vals'] - 2*df['vals'].shift(1) + df['vals'].shift(2)
        df = df[~df['diff_two'].isnull()]
        return df

    def differences_plot(self):
        df = self.differences
        plt.figure()
        plt.title('Differences')
        plt.xlabel('Time')
        plt.plot(df['diff_one'], linestyle='-', color='b')
        plt.plot(df['diff_one'], linestyle='-.', color='grey')
        plt.legend('12')
        plt.show()
        pass

    @classmethod
    def model_forecast_helper(cls, df=None, forecast_date=None, p=0, d=0, q=0, frequency='B') -> tuple:
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
        dict_fcst = {'forecast': forecast[0], 'aic': forecast[1], 'bic': forecast[2]}
        return dict_fcst

    @staticmethod
    def benchmark_params(df, forecast_date, frequency, p, d, q) -> pd.DataFrame:
        performances = []
        for date in utils.iter_date_range(forecast_date[0], forecast_date[1]):
            for p_ in range(p[0], p[1]):
                for d_ in range(d[0], d[1]):
                    for q_ in range(q[0], q[1]):
                        forecast, aic, bic = ARIMA.model_forecast_helper(df=df, forecast_date=date,
                                                                         p=p_, d=d_, q=q_, frequency=frequency)
                        actual = df[df.index == date]
                        performance = {'forecast_date': date,
                                       'p': p_,
                                       'd': d_,
                                       'q': q_,
                                       'aic': aic,
                                       'bic': bic,
                                       'forecast': forecast,
                                       'actual': actual.values[0][0],
                                       'mse': mean_squared_error(y_true=actual, y_pred=[forecast]),
                                       'mae': mean_absolute_error(y_true=actual, y_pred=[forecast])}
                        performances.append(performance)
        performances = pd.DataFrame(performances)
        return performances

    def benchmark_plot(self):
        performances = self.benchmark_params(df=self.model_df, forecast_date=self.forecast_date,
                                             p=self.p, d=self.d, q=self.q, frequency=self.frequency)
        for var in ['p', 'd', 'q']:
            if performances[var].nunique() > 1:
                plt.title('MSE over ' + var.upper())
                plt.plot(performances.groupby(var).mean()['mse'])
                plt.show()
        pass

    @property
    def optimal_params(self) -> pd.DataFrame:
        performances = self.benchmark_params(df=self.model_df, forecast_date=self.forecast_date,
                                             p=self.p, d=self.d, q=self.q, frequency=self.frequency)
        perf_df = pd.DataFrame(performances).groupby(['p', 'd', 'q']).mean()
        optimal_param = perf_df[perf_df[self.criterion] == perf_df[self.criterion].min()]
        return optimal_param

    @property
    def acf_vals(self):
        acf = sm_tools.acf(self.series.head(self.limit), alpha=self.alpha, fft=False)
        return acf

    def acf_plot(self):
        acf = self.acf_vals
        plt.figure()
        plt.title('ACF of ' + self.series.name)
        plt.xlabel('Lags')
        plt.plot(acf[0], linestyle='-', color='b')
        plt.plot(acf[1][:, 0], linestyle='--', color='grey')
        plt.plot(acf[1][:, 1], linestyle='--', color='grey')
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
    ARIMA(series=df['open'],
          datetimes=df['market_datetime'], p=(1, 4), d=(0, 1), q=(0, 1),
          forecast_date=('2019-08-05', '2019-08-09')).benchmark_plot()
