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

    @staticmethod
    def model_calculator(current_stock_price,
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

    def delta(self, stock_price):
        # the rate of change in an options value as the underlying changes
        option_price = self.model_calculator(current_stock_price=stock_price,
                                             strike_price=self.strike_price,
                                             risk_free_rate=self.risk_free_rate,
                                             time_to_maturity=self.time_to_maturity,
                                             volatility=self.volatility)
        return option_price

    def gamma(self, steps=30):
        # the rate of change in an options value as the delta changes
        diff = self.delta(steps=steps)
        diff = diff-diff.shift(-1)
        diff = -diff[~diff.isna()]
        return diff

    def ro(self, rate):
        # the rate of change in an options value as the risk free rate changes
        option_price = self.model_calculator(current_stock_price=self.current_stock_price,
                                             strike_price=self.strike_price,
                                             risk_free_rate=rate,
                                             time_to_maturity=self.time_to_maturity,
                                             volatility=self.volatility)
        return option_price

    def theta(self, day):
        # the rate of change in an options value as the time to maturity changes
        option_price = self.model_calculator(current_stock_price=self.current_stock_price,
                                             strike_price=self.strike_price,
                                             risk_free_rate=self.risk_free_rate,
                                             time_to_maturity=day,
                                             volatility=self.volatility)
        return option_price

    def vega(self, vol):
        # the rate of change in an options value as volatility changes
        option_price = self.model_calculator(current_stock_price=self.current_stock_price,
                                             strike_price=self.strike_price,
                                             risk_free_rate=self.risk_free_rate,
                                             time_to_maturity=self.time_to_maturity,
                                             volatility=vol)
        return option_price

    def get_greek(self, greek=None, steps=30):
        daily_diffs = []
        percent_down = (100 - steps) / 100
        percent_up = (100 + steps) / 100

        if greek == 'delta':
            param = self.current_stock_price
            calculator = self.delta
        elif greek == 'ro':
            param = self.risk_free_rate
            calculator = self.ro
        elif greek == 'theta':
            param = self.time_to_maturity
            calculator = self.theta
        elif greek == 'vega':
            param = self.volatility
            calculator = self.vega

        vals = np.linspace((percent_down * param),
                           ((1 + percent_up) * param),
                           steps * 2 + 1)

        for val in vals:
            option_price = calculator(val)
            daily_diffs.append((val, option_price))
        df = pd.DataFrame(daily_diffs, columns=['val', 'option_price'])
        diff = df['option_price'] - df['option_price'].shift(-1)
        diff = -diff[~diff.isna()]
        diff.index = diff.index - steps
        return diff


if __name__ == '__main__':
    var = BlackScholes(
        current_stock_price=356.59,
        strike_price=385,
        risk_free_rate=1.025,
        days_to_maturity=16,
        volatility=.5).get_greek(greek='delta', steps=10)
    print(var)
