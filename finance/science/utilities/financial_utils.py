import numpy as np
import pandas as pd
import scipy.stats as stats
import scipy.optimize as opt


class BlackScholes:
    def __init__(
            self,
            current_option_price=0,
            stock_price=0,
            strike=0,
            risk_free_rate=0,
            days_to_maturity=0,
            volatility=0,
            is_call=True,
    ):
        self.current_option_price = current_option_price
        self.stock = stock_price
        self.strike = strike
        self.risk_free_rate = risk_free_rate
        self.time = days_to_maturity/365
        self.volatility = volatility
        self.is_call = is_call

    @staticmethod
    def option_price_calculator(
            stock,
            strike,
            risk_free_rate,
            time,
            volatility,
            is_call,
    ):
        d1 = (np.log(stock / strike) + (risk_free_rate + (volatility ** 2) / 2) * time) / (volatility * np.sqrt(time))
        d2 = d1 - volatility * np.sqrt(time)

        # If put, flip sign
        if not is_call:
            d1 = -d1
            d2 = -d2

        d1_p = stats.norm.cdf(d1, 0, 1)
        d2_p = stats.norm.cdf(d2, 0, 1)

        stock_discounted = stock * d1_p
        strike_discounted = strike * (np.e ** (-risk_free_rate * time)) * d2_p

        option = stock_discounted - strike_discounted if is_call else strike_discounted - stock_discounted
        return option

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

    def implied_volatility_helper(self, volatility_guess):
        _option_price = self.option_price_calculator(
            stock=self.stock,
            strike=self.strike,
            risk_free_rate=self.risk_free_rate,
            time=self.time,
            volatility=volatility_guess,
            is_call=self.is_call,
        )
        diff = _option_price - self.current_option_price
        return diff

    @property
    def implied_volatility(self, lower_bound=-15, upper_bound=15):
        implied_volatility = opt.brentq(
            self.implied_volatility_helper,
            lower_bound,
            upper_bound,
        )
        return implied_volatility

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

    # TODO: Correctly code gamma
    def gamma(self):
        """the rate of change in an options value as the delta changes"""
        pass

    def ro(self, risk_free_rate):
        """the rate of change in an options value as the risk free rate changes"""
        option_price = self.option_price_calculator(
            stock=self.stock,
            strike=self.strike,
            risk_free_rate=risk_free_rate,
            time=self.time,
            volatility=self.volatility,
            is_call=self.is_call,
        )
        return option_price

    def theta(self, time):
        """the rate of change in an options value as the time to maturity changes"""
        option_price = self.option_price_calculator(
            stock=self.stock,
            strike=self.strike,
            risk_free_rate=self.risk_free_rate,
            time=time,
            volatility=self.volatility,
            is_call=self.is_call,
        )
        return option_price

    def vega(self, volatility):
        """the rate of change in an options value as volatility changes"""
        option_price = self.option_price_calculator(
            stock=self.stock,
            strike=self.strike,
            risk_free_rate=self.risk_free_rate,
            time=self.time,
            volatility=volatility,
            is_call=self.is_call,
        )
        return option_price

    def get_greek(self, greek=None, steps=30):
        """wrapper to calculate greeks"""
        daily_diffs = []
        percent_down = (100 - steps) / 100
        percent_up = (100 + steps) / 100

        funcs = {
            'delta': {'param': self.stock, 'func': self.delta},
            'ro': {'param': self.risk_free_rate, 'func': self.ro},
            'theta': {'param': self.time, 'func': self.theta},
            'vega': {'param': self.volatility, 'func': self.vega}
        }

        # Percent changes in greek
        vals = np.linspace(
            (percent_down * funcs.get(greek).get('param')),
            (percent_up * funcs.get(greek).get('param')),
            steps * 2 + 1
        )

        for val in vals:
            option_price = funcs.get(greek).get('func')(val)
            daily_diffs.append((val, option_price))

        df = pd.DataFrame(daily_diffs, columns=['val', 'option_price'])
        df[greek] = (df['option_price'].shift(-1)-df['option_price'])/(df['val'].shift(-1)-df['val'])
        df.index = df.index - steps
        df = df[~df.isna()]
        return df[greek]
