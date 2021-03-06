"""options utils"""
import numpy as np
import pandas as pd
import scipy.stats as stats
import scipy.optimize as opt


class BlackScholes:
    def __init__(
            self,
            current_option_price=0,
            stock=0,
            strike=0,
            days_to_maturity=0,
            volatility=0,
            risk_free_rate=0,
            carry_cost=0,
            is_call=True,
            is_future=False,
    ):
        self.current_option_price = current_option_price
        self.stock = stock
        self.strike = strike
        self.time = days_to_maturity / 365
        self.volatility = volatility
        self.risk_free_rate = risk_free_rate
        self.carry_cost = carry_cost
        self.is_call = is_call
        self.is_future = is_future

    @staticmethod
    def calculate_d1(stock, strike, carry_cost, risk_free_rate, volatility, time, is_future):
        b = carry_cost if is_future else risk_free_rate
        d1 = (np.log(stock / strike) + (b + (volatility ** 2) / 2) * time) / (volatility * np.sqrt(time))
        return d1

    @staticmethod
    def calculate_d2(d1, volatility, time):
        d2 = d1 - volatility * np.sqrt(time)
        return d2

    @property
    def d1(self):
        d1 = self.calculate_d1(
            stock=self.stock,
            strike=self.strike,
            time=self.time,
            volatility=self.volatility,
            risk_free_rate=self.risk_free_rate,
            carry_cost=self.carry_cost,
            is_future=self.is_future,
        )
        return d1

    @property
    def d2(self):
        d2 = self.calculate_d2(
            d1=self.d1,
            volatility=self.volatility,
            time=self.time,
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

    def calculate_option_price(
            self,
            stock,
            strike,
            carry_cost,
            risk_free_rate,
            time,
            volatility,
            is_call,
            is_future,
    ):
        """
        Calculate option price
        Note, futures options will not calculate properly in all scenarios
        """
        d1 = self.calculate_d1(
            stock=stock,
            strike=strike,
            time=time,
            volatility=volatility,
            risk_free_rate=risk_free_rate,
            carry_cost=carry_cost,
            is_future=is_future,
        )
        d2 = self.calculate_d2(
            d1=d1,
            volatility=volatility,
            time=time,
        )

        if not is_call:
            d1 = -d1
            d2 = -d2

        d1_p = self._cumulative_density_function(d1)
        d2_p = self._cumulative_density_function(d2)

        stock_discounted = stock * d1_p
        strike_discounted = strike * (np.exp(-risk_free_rate * time)) * d2_p

        option = stock_discounted - strike_discounted if is_call else strike_discounted - stock_discounted
        return option

    @property
    def option_price(self):
        """Given initialized parameters, calculate an options price"""
        option_price = self.calculate_option_price(
            stock=self.stock,
            strike=self.strike,
            time=self.time,
            volatility=self.volatility,
            risk_free_rate=self.risk_free_rate,
            carry_cost=self.carry_cost,
            is_call=self.is_call,
            is_future=self.is_future,
        )
        return option_price

    def calculate_implied_volatility(self, volatility_guess):
        option_price = self.calculate_option_price(
            stock=self.stock,
            strike=self.strike,
            time=self.time,
            volatility=volatility_guess,
            risk_free_rate=self.risk_free_rate,
            carry_cost=self.carry_cost,
            is_call=self.is_call,
            is_future=self.is_future,
        )
        diff = option_price - self.current_option_price
        return diff

    @property
    def implied_volatility(self, lower_bound=-15, upper_bound=15):
        """
        Brentq root finding for implied volatility
        """
        try:
            implied_volatility = opt.brentq(
                self.calculate_implied_volatility,
                lower_bound,
                upper_bound,
                xtol=1e-15,
                rtol=1e-15,
                maxiter=1000,
            )
            implied_volatility = max(round(implied_volatility, 6), 0)
        except ValueError as e:
            print(f'Filling null to workaround {e}')
            implied_volatility = None
        return implied_volatility

    @staticmethod
    def _implied_volatility_seed(stock, strike, risk_free_rate, time):
        """
        Manaster-Koehler implied volatility seed
            sqrt(abs(ln(S/X) + rT) * 2/T)
        """
        vol = np.sqrt(abs(np.log(stock/strike) + risk_free_rate * time) * 2 / time)
        return vol

    @property
    def _implied_volatility(
            self,
            error=.00001,
            max_iterations=100,
            estimated_option_price=0,
    ) -> float:
        """Alternative implied volatility calculation, drawn from Haug"""
        volatility_guess = self._implied_volatility_seed(
            stock=self.stock,
            strike=self.strike,
            risk_free_rate=self.risk_free_rate,
            time=self.time,
        )
        n = 0
        while abs(self.current_option_price - estimated_option_price) > error and n < max_iterations:
            estimated_option_price = self.calculate_option_price(
                stock=self.stock,
                strike=self.strike,
                time=self.time,
                volatility=volatility_guess,
                risk_free_rate=self.risk_free_rate,
                carry_cost=self.carry_cost,
                is_call=self.is_call,
                is_future=self.is_future,
             )
            vega = self.calculate_vega(
                stock=self.stock,
                strike=self.strike,
                time=self.time,
                volatility=volatility_guess,
                risk_free_rate=self.risk_free_rate,
                carry_cost=self.carry_cost,
                is_future=self.is_future,
            )
            volatility_guess = volatility_guess - (
                    (estimated_option_price - self.current_option_price)/vega
            )
            n += 1
        return volatility_guess

    """
    Greeks
    """

    @property
    def delta(self):
        """The rate of change in an options value as the underlying changes"""
        n = 0 if self.is_call else 1
        delta = np.exp(-self.risk_free_rate * self.time) * (self._cumulative_density_function(self.d1) - n)
        return delta

    @property
    def gamma(self):
        """The rate of change in an options value as the delta changes"""
        n = self._probability_density_function(self.d1)
        gamma = (
                (n * np.exp((self.carry_cost - self.risk_free_rate) * self.time))
                / (self.stock * self.volatility * np.sqrt(self.time))
        )
        return gamma

    @property
    def rho(self):
        """The rate of change in an options value as the risk free rate changes"""
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
        return rho

    @property
    def theta(self):
        """The rate of change in an options value as the time to maturity changes"""
        left = -(
                (self.stock * np.exp((self.carry_cost - self.risk_free_rate) * self.time)
                 * self._probability_density_function(self.d1) * self.volatility)
                / (2 * np.sqrt(self.time))
        )
        middle = (
                (self.carry_cost - self.risk_free_rate)
                * self.stock * np.exp((self.carry_cost - self.risk_free_rate) * self.time)
                * self._cumulative_density_function(-self.d1)
        )
        right = (
                self.risk_free_rate * self.strike * np.exp(-self.risk_free_rate * self.time)
                * self._cumulative_density_function(-self.d2)
        )
        theta = left + middle + right
        return theta

    def calculate_vega(
            self,
            stock,
            strike,
            time,
            volatility,
            risk_free_rate,
            carry_cost,
            is_future,
    ):
        d1 = self.calculate_d1(
            stock=stock,
            strike=strike,
            time=time,
            volatility=volatility,
            risk_free_rate=risk_free_rate,
            carry_cost=carry_cost,
            is_future=is_future,
        )
        vega = (
                stock * np.exp((carry_cost - risk_free_rate) * time)
                * self._probability_density_function(d1) * np.sqrt(time)
        )
        return vega

    @property
    def vega(self):
        """The rate of change in an options value as volatility changes"""
        vega = self.calculate_vega(
            stock=self.stock,
            strike=self.strike,
            time=self.time,
            volatility=self.volatility,
            risk_free_rate=self.risk_free_rate,
            carry_cost=self.carry_cost,
            is_future=self.is_future,
        )
        return vega

    @property
    def risk_neutral_probability(self):
        """
        The risk neutral probability of the option, which is the
        probablility that the option finishes in the money
            - For calls, N(D2)
            - For puts, N(-D2)
        """
        x = self.d2 if self.is_call else -self.d2
        rnp = self._cumulative_density_function(x)
        return rnp


def smooth_first_order_difference(
        df: pd.DataFrame,
        degree: int = 2,
) -> pd.DataFrame:
    """
    Smooth the first order differences of an option chain. Recall that an
    option's first order difference is its cumulative density function.
    """
    p = np.polyfit(
        x=df['strike'],
        y=df['first_order_difference'],
        deg=degree,
    )
    df['smoothed_first_order_difference'] = np.polyval(p, df['strike'])
    df.loc[df['smoothed_first_order_difference'] < 0, 'smoothed_first_order_difference'] = 0
    df.loc[df['smoothed_first_order_difference'] > 1, 'smoothed_first_order_difference'] = 1
    df['probability_of_profit'] = 1 - df['smoothed_first_order_difference']
    return df
