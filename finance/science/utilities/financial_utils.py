import numpy as np
import pandas as pd
import scipy.stats as stats
import scipy.optimize as opt


class BlackScholes:
    def __init__(self,
                 current_option_price=0,
                 stock_price=0,
                 strike=0,
                 risk_free_rate=0,
                 days_to_maturity=0,
                 volatility=0,
                 is_call=True):
        self.current_option_price = current_option_price
        self.stock = stock_price
        self.strike = strike
        self.risk_free_rate = risk_free_rate
        self.time = days_to_maturity/365
        self.volatility = volatility
        self.is_call = is_call

    """
    Option prices
    """
    @staticmethod
    def option_price_calculator(
            stock,
            strike,
            risk_free_rate,
            time,
            volatility,
            is_call,
    ):

        d_one = (np.log(current_stock_price/strike_price) + (risk_free_rate+(volatility**2/2))*time_to_maturity) / (volatility * np.sqrt(time_to_maturity))
        d_two = d_one - volatility * np.sqrt(time_to_maturity)

        if is_call:
            option_price = current_stock_price * stats.norm.cdf(d_one, 0, 1) \
                           - strike_price * np.exp(1)**(-risk_free_rate*time_to_maturity) \
                           * stats.norm.cdf(d_two, 0, 1)
        else:
            option_price = strike_price * np.exp(1) ** (-risk_free_rate * time_to_maturity) \
                           * stats.norm.cdf(-d_two, 0, 1) \
                           - current_stock_price * stats.norm.cdf(-d_one, 0, 1)
        return option_price

    @property
    def option_price(self):
        option_price = self.option_price_calculator(
            stock=self.stock,
            strike=self.strike,
            risk_free_rate=self.risk_free_rate,
            time=self.time,
            volatility=self.volatility,
            is_call=self.is_call,
        )
        return option_price

    """
    Implied volatility
    Requires helper since brentq will only work for one input
    """
    def implied_volatility_helper(self, volatility_guess):
        est_option_price = self.option_price_calculator(
            stock=self.stock,
            strike=self.strike,
            risk_free_rate=self.risk_free_rate,
            time=self.time,
            volatility=volatility_guess,
            is_call=self.is_call,
        )
        diff = est_option_price - self.current_option_price
        return diff

    @property
    def implied_volatility(self, lower_bound=-15, upper_bound=15):
        imp_vol = opt.brentq(self.implied_volatility_helper, lower_bound, upper_bound)
        return imp_vol

    """
    Greeks
    """
    def delta(self, stock_price):
        """the rate of change in an options value as the underlying changes"""
        option_price = self.option_price_calculator(
            stock=stock_price,
            strike=self.strike,
            risk_free_rate=self.risk_free_rate,
            time=self.time,
            volatility=self.volatility,
            is_call=self.is_call,
        )
        return option_price

    #TODO
    def gamma(self):
        """the rate of change in an options value as the delta changes"""
        pass

    def ro(self, rate):
        """the rate of change in an options value as the risk free rate changes"""
        option_price = self.option_price_calculator(
            stock=self.stock,
            strike=self.strike,
            risk_free_rate=rate,
            time=self.time,
            volatility=self.volatility,
            is_call=self.is_call,
        )
        return option_price

    def theta(self, day):
        """the rate of change in an options value as the time to maturity changes"""
        option_price = self.option_price_calculator(
            stock=self.stock,
            strike=self.strike,
            risk_free_rate=self.risk_free_rate,
            time=day,
            volatility=self.volatility,
            is_call=self.is_call,
        )
        return option_price

    def vega(self, vol):
        """the rate of change in an options value as volatility changes"""
        option_price = self.option_price_calculator(
            stock=self.stock,
            strike=self.strike,
            risk_free_rate=self.risk_free_rate,
            time=self.time,
            volatility=vol,
            is_call=self.is_call,
        )
        return option_price

    def get_greek(self, greek=None, steps=30):
        """wrapper to calculate greeks"""
        daily_diffs = []
        percent_down = (100 - steps) / 100
        percent_up = (100 + steps) / 100

        funcs = {
            'delta': {'param': self.current_stock_price, 'func': self.delta},
            'ro': {'param': self.risk_free_rate, 'func': self.ro},
            'theta': {'param': self.time_to_maturity, 'func': self.theta},
            'vega': {'param': self.volatility, 'func': self.vega}
        }

        vals = np.linspace((percent_down * funcs.get(greek).get('param')),
                           ((1 + percent_up) * funcs.get(greek).get('param')),
                           steps * 2 + 1)

        for val in vals:
            option_price = funcs.get(greek).get('func')(val)
            daily_diffs.append((val, option_price))
        df = pd.DataFrame(daily_diffs, columns=['val', 'option_price'])
        diff = df['option_price'] - df['option_price'].shift(-1)
        diff = -diff[~diff.isna()]
        diff.index = diff.index - steps
        return diff
