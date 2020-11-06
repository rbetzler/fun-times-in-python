import datetime
import requests
from typing import NamedTuple

from finance.trading import access
from finance.utilities import utils


_ACCOUNT = utils.retrieve_secret('TD_PRIMARY_ACCOUNT')
_ACCESS_TOKEN: str = access.get_access_token()

_ACCOUNTS_URL = r'https://api.tdameritrade.com/v1/accounts'
_ORDER_URL = 'https://api.tdameritrade.com/v1/accounts/{account}/orders'

_RAW_DT_FMT = '%Y-%m-%dT%H:%M:%S+0000'
_DT_FMT = '%Y-%m-%d %H:%M:%S'


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


def _get_headers(access_token: str) -> dict:
    headers = {'Authorization': f"Bearer {access_token}"}
    return headers


def get_accounts(
        access_token: str = _ACCESS_TOKEN,
        url: str = _ACCOUNTS_URL,
) -> dict:
    r = requests.get(url=url, headers=_get_headers(access_token))
    j = r.json()
    return j


def get_orders(
        account: str = _ACCOUNT,
        access_token: str = _ACCESS_TOKEN,
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
        access_token: str = _ACCESS_TOKEN,
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
        access_token: str = _ACCESS_TOKEN,
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


if __name__ == '__main__':
    orders = [
        TDOrder(symbol='AA', price='1.5', quantity=2),
        TDOrder(symbol='AIG', price='1.5', quantity=3),
    ]
    for order in orders:
        place_order(order)

    orders = get_orders()
    for order in orders:
        if order.status == 'WORKING':
            cancel_order(order_id=order.id)
