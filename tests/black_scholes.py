import unittest
from utilities import options_utils

D1 = -0.3253
D2 = -0.4753
CDF = 0.3725
CDF_2 = 0.3173
CALL = 2.1334
DELTA_CALL = 0.6629
DELTA_PUT = -0.2884
FUTURE_DELTA_CALL = 0.5946
FUTURE_DELTA_PUT = -0.3566
GAMMA = 0.0278
VEGA = 18.5027
THETA_PUT = -31.1924
RHO = 38.7325
VOLATILITY = .2202
VOLATILITY_2 = .217
VOLATILITY_3 = .231


class BlackScholes(unittest.TestCase):
    """Unit tests for the Black Scholes calculator"""

    def test_d1(self):
        """
        Whether we calculate d1 correctly
        """
        raw_d1 = options_utils.BlackScholes().calculate_d1(
            stock=60,
            strike=65,
            carry_cost=0,
            risk_free_rate=.08,
            volatility=.3,
            time=.25,
            is_future=False,
        )
        d1 = round(raw_d1, 4)
        self.assertEqual(D1, d1)

    def test_d2(self):
        """
        Whether we calculate d2 correctly
        """
        raw_d2 = options_utils.BlackScholes().calculate_d2(
            d1=D1,
            volatility=.3,
            time=.25,
        )
        d2 = round(raw_d2, 4)
        self.assertEqual(D2, d2)

    def test_cdf(self):
        """
        Whether we calculate cdf correctly
        """
        raw_cdf = options_utils.BlackScholes()._cumulative_density_function(
            d=D1,
        )
        cdf = round(raw_cdf, 4)
        self.assertEqual(CDF, cdf)

        raw_cdf_2 = options_utils.BlackScholes()._cumulative_density_function(
            d=D2,
        )
        cdf_2 = round(raw_cdf_2, 4)
        self.assertEqual(CDF_2, cdf_2)

    def test_call_price(self):
        """
        Whether we calculate call option prices correctly
        """
        raw_call = options_utils.BlackScholes().calculate_option_price(
            stock=60,
            strike=65,
            carry_cost=0,
            risk_free_rate=.08,
            volatility=.3,
            time=.25,
            is_call=True,
            is_future=False,
        )
        call = round(raw_call, 4)
        self.assertEqual(CALL, call)

    def test_option_price_symmetry(self):
        """
        Whether we calculate puts option prices correctly

        A call and a put with opposite stock prices, strikes,
        and vol will be equal to one another:
            C(S, X, Sigma) = P(-S, -X, -Sigma)

        Consequently, to calculate an put option price, one can
        use the call option formula with a few negative parameters
        """
        raw_call = options_utils.BlackScholes().calculate_option_price(
            stock=60,
            strike=65,
            carry_cost=0,
            risk_free_rate=.08,
            volatility=.3,
            time=.25,
            is_call=True,
            is_future=False,
        )
        call = round(raw_call, 4)
        self.assertEqual(CALL, call)

        raw_put = options_utils.BlackScholes().calculate_option_price(
            stock=-60,
            strike=-65,
            carry_cost=0,
            risk_free_rate=.08,
            volatility=-.3,
            time=.25,
            is_call=False,
            is_future=False,
        )
        put = round(raw_put, 4)
        self.assertEqual(call, put)

    def test_delta(self):
        """
        Whether we calculate delta correctly
        """
        raw_delta = options_utils.BlackScholes(
            stock=105,
            strike=100,
            risk_free_rate=0.1,
            volatility=.36,
            days_to_maturity=365/2,
        ).delta
        delta_call = round(raw_delta, 4)
        self.assertEqual(DELTA_CALL, delta_call)

        raw_delta = options_utils.BlackScholes(
            stock=105,
            strike=100,
            risk_free_rate=0.1,
            volatility=.36,
            days_to_maturity=365/2,
            is_call=False,
        ).delta
        delta_put = round(raw_delta, 4)
        self.assertEqual(DELTA_PUT, delta_put)

    def test_delta_futures(self):
        """
        Whether we calculate delta on futures correctly
        Used to match examples in Haug's textbook
        """
        raw_delta = options_utils.BlackScholes(
            stock=105,
            strike=100,
            risk_free_rate=0.1,
            volatility=.36,
            days_to_maturity=365/2,
            is_future=True,
        ).delta
        future_delta_call = round(raw_delta, 4)
        self.assertEqual(FUTURE_DELTA_CALL, future_delta_call)

        raw_delta = options_utils.BlackScholes(
            stock=105,
            strike=100,
            risk_free_rate=0.1,
            volatility=.36,
            days_to_maturity=365/2,
            is_call=False,
            is_future=True,
        ).delta
        future_delta_put = round(raw_delta, 4)
        self.assertEqual(FUTURE_DELTA_PUT, future_delta_put)

    def test_gamma(self):
        """
        Whether we calculate gamma correctly
        """
        raw_gamma = options_utils.BlackScholes(
            stock=55,
            strike=60,
            risk_free_rate=.1,
            carry_cost=.1,
            volatility=.3,
            days_to_maturity=365 * .75,
        ).gamma
        gamma_call = round(raw_gamma, 4)
        self.assertEqual(GAMMA, gamma_call)

        raw_gamma = options_utils.BlackScholes(
            stock=55,
            strike=60,
            risk_free_rate=0.1,
            volatility=.3,
            carry_cost=.1,
            days_to_maturity=365 * .75,
            is_call=False,
        ).gamma
        gamma_put = round(raw_gamma, 4)
        self.assertEqual(GAMMA, gamma_put)

    def test_vega(self):
        """
        Whether we calculate vega correctly
        Haug uses index options, hence is_future
        """
        raw_vega = options_utils.BlackScholes(
            stock=55,
            strike=60,
            risk_free_rate=0.105,
            volatility=.3,
            carry_cost=.0695,
            days_to_maturity=365 * .75,
            is_future=True,
        ).vega
        vega_call = round(raw_vega, 4)
        self.assertEqual(VEGA, vega_call)

        raw_vega = options_utils.BlackScholes(
            stock=55,
            strike=60,
            risk_free_rate=0.105,
            volatility=.3,
            carry_cost=.0695,
            days_to_maturity=365 * .75,
            is_call=False,
            is_future=True,
        ).vega
        vega_put = round(raw_vega, 4)
        self.assertEqual(VEGA, vega_put)

    def test_theta(self):
        """
        Whether we calculate theta correctly
        """
        raw_theta = options_utils.BlackScholes(
            stock=430,
            strike=405,
            risk_free_rate=0.07,
            volatility=.2,
            carry_cost=.02,
            days_to_maturity=365/12,
            is_call=False,
            is_future=True,
        ).theta
        theta_put = round(raw_theta, 4)
        self.assertEqual(THETA_PUT, theta_put)

    def test_rho(self):
        """
        Whether we calculate rho correctly
        """
        raw_rho = options_utils.BlackScholes(
            stock=72,
            strike=75,
            risk_free_rate=0.09,
            volatility=.19,
            carry_cost=.09,
            days_to_maturity=365,
        ).rho
        rho = round(raw_rho, 4)
        self.assertEqual(RHO, rho)

    def test_implied_volatility(self):
        raw_volatility = options_utils.BlackScholes(
            current_option_price=.68,
            stock=48.51,
            strike=48.5,
            risk_free_rate=0.01,
            days_to_maturity=9,
        )._implied_volatility()
        volatility = round(raw_volatility, 4)
        self.assertEqual(VOLATILITY, volatility)

    def test_implied_volatility_2(self):
        raw_volatility = options_utils.BlackScholes(
            current_option_price=.45,
            stock=48.51,
            strike=49,
            risk_free_rate=0.01,
            days_to_maturity=9,
        )._implied_volatility()
        volatility = round(raw_volatility, 3)
        self.assertEqual(VOLATILITY_2, volatility)

    def test_implied_volatility_3(self):
        raw_volatility = options_utils.BlackScholes(
            current_option_price=.61,
            stock=48.51,
            strike=52.5,
            risk_free_rate=0.03,
            days_to_maturity=88 * 5/7,
        )._implied_volatility()
        volatility = round(raw_volatility, 3)
        self.assertEqual(VOLATILITY_3, volatility)


if __name__ == '__main__':
    unittest.main()
