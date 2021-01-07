import datetime
import requests
from typing import List, NamedTuple
from trading import access
from utilities import utils

_ACCOUNT = utils.retrieve_secret('TD_PRIMARY_ACCOUNT')
_ACCOUNTS_URL = r'https://api.tdameritrade.com/v1/accounts?fields=positions'
_ORDER_URL = 'https://api.tdameritrade.com/v1/accounts/{account}/orders'
_RAW_DT_FMT = '%Y-%m-%dT%H:%M:%S+0000'
_DT_FMT = '%Y-%m-%d %H:%M:%S'


class TDAccount(NamedTuple):
    id: str
    available_funds: float
    positions: list


class TDOrder(NamedTuple):
    id: str = None
    status: str = None
    symbol: str = None
    price: str = None
    quantity: int = None
    instruction: str = 'BUY'
    asset_type: str = 'EQUITY'
    order_type: str = 'LIMIT'
    account: str = _ACCOUNT
    entered_time: str = datetime.datetime.utcnow().strftime(_DT_FMT)
    closed_time: str = datetime.datetime.utcnow().strftime(_DT_FMT)


class TDPosition(NamedTuple):
    symbol: str
    asset_type: str
    average_price: float
    market_value: float
    net_quantity: int


def _get_headers(access_token: str) -> dict:
    headers = {'Authorization': f"Bearer {access_token}"}
    return headers


class TDAccounts:
    @staticmethod
    def get_accounts(
            access_token: str = access.get_access_token(),
            url: str = _ACCOUNTS_URL,
    ) -> List[TDAccount]:
        """Get available funds and positions for each td account"""
        r = requests.get(url=url, headers=_get_headers(access_token))
        j = r.json()
        accounts = []
        for acnt in j:
            a = acnt.get('securitiesAccount')
            balances = a.get('currentBalances')
            available_funds = balances.get('cashAvailableForWithdrawal') or balances.get('availableFundsNonMarginableTrade')
            ps = a.get('positions')
            positions = []
            for p in ps:
                i = p.get('instrument')
                symbol = i.get('symbol')
                asset_type = i.get('assetType')
                average_price = p.get('averagePrice')
                market_value = p.get('marketValue')
                long_quantity = p.get('longQuantity')
                short_quantity = p.get('shortQuantity')
                net_quantity = long_quantity - short_quantity
                position = TDPosition(
                    symbol=symbol,
                    asset_type=asset_type,
                    average_price=average_price,
                    market_value=market_value,
                    net_quantity=net_quantity,
                )
                positions.append(position)
            account = TDAccount(
                id=a.get('accountId'),
                available_funds=available_funds,
                positions=positions,
            )
            accounts.append(account)
        return accounts

    def get_positions(self) -> List[TDPosition]:
        positions = []
        for a in self.get_accounts():
            positions.extend(a.positions)
        return positions


def get_orders(
        account: str = _ACCOUNT,
        access_token: str = access.get_access_token(),
        url: str = _ORDER_URL,
) -> list:

    r = requests.get(
        url=url.format(account=account),
        headers=_get_headers(access_token))
    j = r.json()

    orders = []
    for order in j:
        o = TDOrder(
            id=str(order.get('orderId')),
            status=order.get('status'),
            symbol=order.get('orderLegCollection')[0].get('instrument').get('symbol'),
            price=order.get('price'),
            quantity=order.get('quantity'),
            instruction=order.get('orderLegCollection')[0].get('instruction'),
            asset_type=order.get('orderLegCollection')[0].get('orderLegType'),
            order_type=order.get('orderType'),
            account=order.get('accountId'),
            entered_time=datetime.datetime.strptime(order.get('enteredTime'), _RAW_DT_FMT).strftime(_DT_FMT),
            closed_time=datetime.datetime.strptime(order.get('closeTime'), _RAW_DT_FMT).strftime(_DT_FMT) if order.get('closeTime') else '3000-01-01 01:01:00'
        )
        orders.append(o)
    return orders


def place_order(
        order: TDOrder,
        access_token: str = access.get_access_token(),
        url: str = _ORDER_URL,
):
    json = {
        "orderType": order.order_type,
        "session": "NORMAL",
        "price": order.price,
        "duration": "DAY",
        "orderStrategyType": "SINGLE",
        "orderLegCollection": [
            {
                "instruction": order.instruction,
                "quantity": order.quantity,
                "instrument": {
                    "symbol": order.symbol,
                    "assetType": order.asset_type
                }
            }
        ]
    }
    r = requests.post(
        url=url.format(account=order.account),
        headers=_get_headers(access_token),
        json=json)
    if r.status_code in (200, 201):
        print('Successfully placed order for: ' + order.symbol)
    else:
        raise RuntimeError('Failed to place order for: ' + order.symbol)


def cancel_order(
        order_id: str,
        account: str = _ACCOUNT,
        access_token: str = access.get_access_token(),
        url: str = _ORDER_URL,
):
    cancel_url = url.format(account=account) + '/' + order_id
    r = requests.delete(
        url=cancel_url,
        headers=_get_headers(access_token))
    if r.status_code in (200, 201):
        print('Successfully canceled order: ' + order_id)
    else:
        raise RuntimeError('Order was not successfully canceled: ' + order_id)
