import os
import psycopg2
import numpy as np
import pandas as pd
import scipy.stats as stats


class BlackScholes:
    def __init__(self,
                 current_stock_price=0,
                 strike_price=0,
                 risk_free_rate=0,
                 days_to_maturity=0,
                 volatility=0):
        self.current_stock_price = current_stock_price
        self.strike_price = strike_price
        self.risk_free_rate = np.log(risk_free_rate)
        # time in days
        self.time_to_maturity = days_to_maturity/365
        self.volatility = volatility

    def model_calculator(self,
                         current_stock_price,
                         strike_price,
                         risk_free_rate,
                         time_to_maturity,
                         volatility):
        d_one = (np.log(current_stock_price/strike_price) + (risk_free_rate+(volatility**2/2))*time_to_maturity) \
                / (volatility * np.sqrt(time_to_maturity))
        d_two = d_one - volatility * np.sqrt(time_to_maturity)
        option_price = current_stock_price * stats.norm.cdf(d_one, 0, 1) \
                       - strike_price * np.exp(1)**(-risk_free_rate*time_to_maturity) \
                       * stats.norm.cdf(d_two, 0, 1)
        return option_price

    @property
    def option_price(self):
        option_price = self.model_calculator(current_stock_price=self.current_stock_price,
                                             strike_price=self.strike_price,
                                             risk_free_rate=self.risk_free_rate,
                                             time_to_maturity=self.time_to_maturity,
                                             volatility=self.volatility)
        return option_price

    def delta(self, steps=30):
        # the rate of change in an options value as the underlying changes
        daily_diffs = []
        percent_diff = (100-steps)/100
        stock_prices = np.linspace((percent_diff * self.current_stock_price),
                                   ((1 + percent_diff) * self.current_stock_price),
                                   steps * 2)
        for stock_price in stock_prices:
            option_price = self.model_calculator(current_stock_price=stock_price,
                                                 strike_price=self.strike_price,
                                                 risk_free_rate=self.risk_free_rate,
                                                 time_to_maturity=self.time_to_maturity,
                                                 volatility=self.volatility)
            daily_diffs.append((stock_price, option_price))
        df = pd.DataFrame(daily_diffs, columns=['stock_price', 'option_price'])
        diff = df['option_price']-df['option_price'].shift(-1)
        diff = -diff[~diff.isna()]
        diff.index = diff.index - steps
        return diff

    def gamma(self, steps=30):
        # the rate of change in an options value as the delta changes
        diff = self.delta(steps=steps)
        diff = diff-diff.shift(-1)
        diff = -diff[~diff.isna()]
        return diff

    def ro(self, steps=30):
        # the rate of change in an options value as the risk free rate changes
        daily_diffs = []
        percent_diff = (100-steps)/100
        rates = np.linspace((percent_diff * self.risk_free_rate),
                                   ((1 + percent_diff) * self.risk_free_rate),
                                   steps * 2)
        for rate in rates:
            option_price = self.model_calculator(current_stock_price=self.current_stock_price,
                                                 strike_price=self.strike_price,
                                                 risk_free_rate=rate,
                                                 time_to_maturity=self.time_to_maturity,
                                                 volatility=self.volatility)
            daily_diffs.append((rate, option_price))
        df = pd.DataFrame(daily_diffs, columns=['rate', 'option_price'])
        diff = df['option_price']-df['option_price'].shift(-1)
        diff = -diff[~diff.isna()]
        diff.index = diff.index - steps
        return diff

    def theta(self, steps=30):
        # the rate of change in an options value as time changes
        daily_diffs = []
        percent_diff = (100 - steps) / 100
        days = np.linspace((percent_diff * self.time_to_maturity),
                                   ((1 + percent_diff) * self.time_to_maturity),
                                   steps * 2)
        for day in days:
            option_price = self.model_calculator(current_stock_price=self.current_stock_price,
                                                 strike_price=self.strike_price,
                                                 risk_free_rate=self.risk_free_rate,
                                                 time_to_maturity=day,
                                                 volatility=self.volatility)
            daily_diffs.append((day, option_price))
        df = pd.DataFrame(daily_diffs, columns=['day', 'option_price'])
        diff = df['option_price']-df['option_price'].shift(-1)
        diff = -diff[~diff.isna()]
        diff.index = diff.index - steps
        return diff

    def vega(self, steps=30):
        # the rate of change in an options value as volatility changes
        daily_diffs = []
        percent_diff = (100 - steps) / 100
        vols = np.linspace((percent_diff * self.volatility),
                                   ((1 + percent_diff) * self.volatility),
                                   steps * 2)
        for vol in vols:
            option_price = self.model_calculator(current_stock_price=self.current_stock_price,
                                                 strike_price=self.strike_price,
                                                 risk_free_rate=self.risk_free_rate,
                                                 time_to_maturity=self.time_to_maturity,
                                                 volatility=vol)
            daily_diffs.append((vol, option_price))
        df = pd.DataFrame(daily_diffs, columns=['vol', 'option_price'])
        diff = df['option_price']-df['option_price'].shift(-1)
        diff = -diff[~diff.isna()]
        diff.index = diff.index - steps
        return diff


# if __name__ == '__main__':
#     var = BlackScholes(
#         current_stock_price=254,
#         strike_price=275,
#         risk_free_rate=.34,
#         time_to_maturity=(2/365),
#         volatility=1).delta(steps=10)
#     print(var)