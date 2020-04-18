import requests
from finance.trading import access
from finance.utilities import utils

_ACCOUNT = utils.retrieve_secret('TD_PRIMARY_ACCOUNT')
_ACCESS_TOKEN: str = access.get_access_token()

_ACCOUNTS_URL = r'https://api.tdameritrade.com/v1/accounts'
_ORDER_URL = 'https://api.tdameritrade.com/v1/accounts/{account}/orders'


def _get_headers(access_token: str):
    headers = {'Authorization': f"Bearer {access_token}"}
    return headers


def get_accounts(
        access_token: str=_ACCESS_TOKEN,
        url: str=_ACCOUNTS_URL,
):
    r = requests.get(url=url, headers=_get_headers(access_token))
    j = r.json()
    return j


def get_orders(
        account: str=_ACCOUNT,
        access_token: str=_ACCESS_TOKEN,
        url: str=_ORDER_URL,
):
    r = requests.get(
        url=url.format(account=account),
        headers=_get_headers(access_token))
    j = r.json()
    return j


def place_order(
        symbol: str='KO',
        price: str='0',
        quantity: int=0,
        instruction: str='BUY',
        asset_type: str='EQUITY',
        order_type: str='LIMIT',
        account: str=_ACCOUNT,
        access_token: str=_ACCESS_TOKEN,
        url: str=_ORDER_URL,
):
    json = {
        "orderType": order_type,
        "session": "NORMAL",
        "price": price,
        "duration": "DAY",
        "orderStrategyType": "SINGLE",
        "orderLegCollection": [
            {
                "instruction": instruction,
                "quantity": quantity,
                "instrument": {
                    "symbol": symbol,
                    "assetType": asset_type
                }
            }
        ]
    }
    r = requests.post(
        url=url.format(account=account),
        headers=_get_headers(access_token),
        json=json)
    return r


def cancel_order(
        order_id: str,
        account: str=_ACCOUNT,
        access_token: str=_ACCESS_TOKEN,
        url: str=_ORDER_URL,
):
    cancel_url = url.format(account=account) + '/' + order_id
    r = requests.delete(
        url=cancel_url,
        headers=_get_headers(access_token))
    j = r.json()
    return j


if __name__ == '__main__':
    res = get_orders()
    res
