import numpy as np
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
        self.time = days_to_maturity / 365
        self.volatility = volatility
        self.is_call = is_call

    @staticmethod
    def _d1(stock, strike, risk_free_rate, volatility, time, is_call):
        d1 = (np.log(stock / strike) + (risk_free_rate + (volatility ** 2) / 2) * time) / (volatility * np.sqrt(time))
        if not is_call:
            d1 = -d1
        return d1

    @property
    def d1(self):
        d1 = self._d1(
            stock=self.stock,
            strike=self.strike,
            risk_free_rate=self.risk_free_rate,
            volatility=self.volatility,
            time=self.time,
            is_call=self.is_call,
        )
        return d1

    @staticmethod
    def _d2(d1, volatility, time, is_call):
        d2 = d1 - volatility * np.sqrt(time)
        if not is_call:
            d2 = -d2
        return d2

    @property
    def d2(self):
        d2 = self._d2(
            d1=self.d1,
            volatility=self.volatility,
            time=self.time,
            is_call=self.is_call,
        )
        return d2

    @staticmethod
    def _probability_density_function(d, mean=0, standard_deviation=1):
        pndf = stats.norm.pdf(d, mean, standard_deviation)
        return pndf

    @staticmethod
    def _cumulative_density_function(d, mean=0, standard_deviation=1):
        cndf = stats.norm.cdf(d, mean, standard_deviation)
        return cndf

    def _calculate_option_price(
            self,
            stock,
            strike,
            risk_free_rate,
            time,
            volatility,
            is_call,
    ):
        d1 = self._d1(
            stock=stock,
            strike=strike,
            risk_free_rate=risk_free_rate,
            volatility=volatility,
            time=time,
            is_call=is_call
        )
        d2 = self._d2(
            d1=d1,
            volatility=volatility,
            time=time,
            is_call=is_call
        )

        d1_p = self._cumulative_density_function(d1)
        d2_p = self._cumulative_density_function(d2)

        stock_discounted = stock * d1_p
        strike_discounted = strike * (np.exp(-risk_free_rate * time)) * d2_p

        option = stock_discounted - strike_discounted if is_call else strike_discounted - stock_discounted
        return option

    @property
    def option_price(self):
        option_price = self._calculate_option_price(
            stock=self.stock,
            strike=self.strike,
            risk_free_rate=self.risk_free_rate,
            time=self.time,
            volatility=self.volatility,
            is_call=self.is_call,
        )
        return option_price

    def _implied_volatility(self, volatility_guess):
        _option_price = self._calculate_option_price(
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
            self._implied_volatility,
            lower_bound,
            upper_bound,
        )
        return implied_volatility

    """
    Greeks
    """

    @property
    def delta(self):
        """the rate of change in an options value as the underlying changes"""
        delta = np.exp(-self.risk_free_rate * self.time) * self._cumulative_density_function(self.d1)
        return delta

    @property
    def gamma(self):
        """the rate of change in an options value as the delta changes"""
        gamma = (
                (self._probability_density_function(self.d1) * np.exp(-self.risk_free_rate * self.time))
                / (self.stock * self.volatility * np.sqrt(self.time))
        )
        return gamma

    @property
    def rho(self):
        """the rate of change in an options value as the risk free rate changes"""
        if self.is_call:
            rho = (
                    self.time * self.strike * np.exp(-self.risk_free_rate * self.time)
                    * self._cumulative_density_function(self.d2)
            )
        else:
            rho = (
                    -self.time * self.strike * np.exp(-self.risk_free_rate * self.time)
                    * self._cumulative_density_function(-self.d2)
            )
        rho = rho / 100
        return rho

    @property
    def theta(self):
        """the rate of change in an options value as the time to maturity changes"""
        left = -(
                (self.stock * np.exp(-self.risk_free_rate * self.time)
                 * self._probability_density_function(self.d1) * self.volatility)
                / (2 * np.sqrt(self.time))
        )
        middle = (
                -self.risk_free_rate * self.stock * np.exp(-self.risk_free_rate * self.time)
                * self._cumulative_density_function(self.d1)
        )
        center = (
                self.risk_free_rate * self.strike * np.exp(-self.risk_free_rate * self.time)
                * self._cumulative_density_function(self.d2)
        )
        theta = left - middle - center
        theta = theta / 100
        return theta

    @property
    def vega(self):
        """the rate of change in an options value as volatility changes"""
        vega = (
                self.stock * np.exp(-self.risk_free_rate * self.time)
                * self._probability_density_function(self.d1) * np.sqrt(self.time)
        )
        vega = vega / 100
        return vega
