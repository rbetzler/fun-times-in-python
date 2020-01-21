import numpy as np
import pandas as pd
import scipy.stats as stats
import scipy.optimize as opt


class BlackScholes:
    def __init__(self,
                 current_option_price=0,
                 current_stock_price=0,
                 strike_price=0,
                 risk_free_rate=0,
                 days_to_maturity=0,
                 volatility=0,
                 call_put=None):
        self.current_option_price = current_option_price
        self.current_stock_price = current_stock_price
        self.strike_price = strike_price
        self.risk_free_rate = np.log(risk_free_rate)
        # time in days
        self.time_to_maturity = days_to_maturity/365
        self.volatility = volatility
        self.call_put = call_put

    """
    Option prices
    """
    @staticmethod
    def option_price_calculator(current_stock_price,
                                strike_price,
                                risk_free_rate,
                                time_to_maturity,
                                volatility,
                                call_put):

        d_one = (np.log(current_stock_price/strike_price) + (risk_free_rate+(volatility**2/2))*time_to_maturity) \
                / (volatility * np.sqrt(time_to_maturity))
        d_two = d_one - volatility * np.sqrt(time_to_maturity)

        if call_put == 'call':
            option_price = current_stock_price * stats.norm.cdf(d_one, 0, 1) \
                           - strike_price * np.exp(1)**(-risk_free_rate*time_to_maturity) \
                           * stats.norm.cdf(d_two, 0, 1)
        elif call_put == 'put':
            option_price = strike_price * np.exp(1) ** (-risk_free_rate * time_to_maturity) \
                           * stats.norm.cdf(-d_two, 0, 1) \
                           - current_stock_price * stats.norm.cdf(-d_one, 0, 1)
        return option_price

    @property
    def option_price(self):
        option_price = self.option_price_calculator(current_stock_price=self.current_stock_price,
                                                    strike_price=self.strike_price,
                                                    risk_free_rate=self.risk_free_rate,
                                                    time_to_maturity=self.time_to_maturity,
                                                    volatility=self.volatility,
                                                    call_put=self.call_put)
        return option_price

    """
    Implied volatility
    Requires helper since brentq will only work for one input
    """
    def implied_volatility_helper(self, volatility_guess):
        est_option_price = self.option_price_calculator(current_stock_price=self.current_stock_price,
                                                        strike_price=self.strike_price,
                                                        risk_free_rate=self.risk_free_rate,
                                                        time_to_maturity=self.time_to_maturity,
                                                        volatility=volatility_guess,
                                                        call_put=self.call_put)
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
        # the rate of change in an options value as the underlying changes
        option_price = self.option_price_calculator(current_stock_price=stock_price,
                                                    strike_price=self.strike_price,
                                                    risk_free_rate=self.risk_free_rate,
                                                    time_to_maturity=self.time_to_maturity,
                                                    volatility=self.volatility,
                                                    call_put=self.call_put)
        return option_price

    def gamma(self):
        # the rate of change in an options value as the delta changes
        pass

    def ro(self, rate):
        # the rate of change in an options value as the risk free rate changes
        option_price = self.option_price_calculator(current_stock_price=self.current_stock_price,
                                                    strike_price=self.strike_price,
                                                    risk_free_rate=rate,
                                                    time_to_maturity=self.time_to_maturity,
                                                    volatility=self.volatility,
                                                    call_put=self.call_put)
        return option_price

    def theta(self, day):
        # the rate of change in an options value as the time to maturity changes
        option_price = self.option_price_calculator(current_stock_price=self.current_stock_price,
                                                    strike_price=self.strike_price,
                                                    risk_free_rate=self.risk_free_rate,
                                                    time_to_maturity=day,
                                                    volatility=self.volatility,
                                                    call_put=self.call_put)
        return option_price

    def vega(self, vol):
        # the rate of change in an options value as volatility changes
        option_price = self.option_price_calculator(current_stock_price=self.current_stock_price,
                                                    strike_price=self.strike_price,
                                                    risk_free_rate=self.risk_free_rate,
                                                    time_to_maturity=self.time_to_maturity,
                                                    volatility=vol,
                                                    call_put=self.call_put)
        return option_price

    """
    Greek wrapper
    """
    def get_greek(self, greek=None, steps=30):
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


def annualized_return(start_price, end_price, n_days):
    annual_return = (1+((end_price-start_price)/start_price))**(365/n_days)-1
    return annual_return


def kelly_criterion(predicted_win, predicted_loss, p_win):
    bet_size = (predicted_win * p_win - predicted_loss * (1 - p_win)) / predicted_win
    return bet_size


def get_estimated_loss(df, 
                       profits: str='profits',
                       window_size: int=100,
                       loss: str='median'
                      ):
    df['is_profitable'] = False
    df.loc[df[profits] > 0, 'is_profitable'] = True

    df['profit_rate'] = (df['is_profitable'].rolling(window_size).sum()
                          /df['is_profitable'].rolling(window_size).count())
    
    if loss == 'mean':
        df['estimated_loss'] = abs(df[profits]).rolling(window_size).mean()
    elif loss == 'median':
        df['estimated_loss'] = abs(df[profits]).rolling(window_size).median()

    return df.drop(['is_profitable'], axis=1)


def greedy_kelly(df,
                 level: str='market_datetime',
                 kelly: str='kelly',
                 price: str='open',
                 profits: str='profits',
                 budget: float=1000,
                ):
    df['position'] = 0
    df['n_shares'] = 0

    positions_dfs = []
    for idx, temp in df.groupby(level):
        stocks = temp[temp['kelly'] > 0]
        positions = stocks.loc[stocks.sort_values(by=kelly, ascending=False)[kelly].cumsum() < 1].copy()
        positions['n_shares'] = ((positions[kelly] * budget)/positions[price]).astype(int)
        positions['position'] = positions['n_shares'] * positions[price]
        budget += sum(positions['n_shares'] * positions[profits])
        positions_dfs.append(positions)

    return pd.concat(positions_dfs)


def n_largest_kelly(df,
                    n_stocks: int=3,
                    level: str='market_datetime',
                    kelly: str='kelly',
                    price: str='open',
                    profits: str='profits',
                    budget: float=1000,
                   ):
    df['position'] = 0
    df['n_shares'] = 0

    positions_dfs = []
    for idx, temp in df.groupby(level):
        stocks = temp[temp[kelly] > 0]
        positions = stocks.nlargest(n_stocks, kelly)
        df[kelly] = df[kelly] * (1 / df[kelly].sum())
        positions['n_shares'] = ((positions[kelly] * budget)/positions[price]).astype(int)
        positions['position'] = positions['n_shares'] * positions[price]
        budget += sum(positions['n_shares'] * positions[profits])
        positions_dfs.append(positions)

    return pd.concat(positions_dfs)