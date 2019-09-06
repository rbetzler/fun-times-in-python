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
                 time_to_maturity=0,
                 volatility=0):
        self.current_stock_price = current_stock_price
        self.strike_price = strike_price
        self.risk_free_rate = risk_free_rate
        self.time_to_maturity = time_to_maturity
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
        option_price = current_stock_price * stats.norm.cdf(d_one, 0, 1) - \
                       strike_price * np.exp(1)**(-risk_free_rate*time_to_maturity) \
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

    @property
    def theta(self, steps=30):
        # the rate of change in an options value as time changes
        daily_diffs = []
        days = np.linspace((1/365), self.time_to_maturity, steps)
        for day in days:
            option_price = self.model_calculator(current_stock_price=self.current_stock_price,
                                                 strike_price=self.strike_price,
                                                 risk_free_rate=self.risk_free_rate,
                                                 time_to_maturity=day,
                                                 volatility=self.volatility)
            daily_diffs.append((day, option_price))
        df = pd.DataFrame(daily_diffs, columns=['day', option_price])
        return theta


if __name__ == '__main__':
    t = BlackScholes(
        current_stock_price=254,
        strike_price=275,
        risk_free_rate=.34,
        time_to_maturity=(2/365),
        volatility=1).theta(steps=3)
    print(t)