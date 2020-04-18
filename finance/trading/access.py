import requests
import urllib
from finance.utilities import utils

_AUTH_URL = r'https://api.tdameritrade.com/v1/oauth2/token'
_ACCOUNTS_URL = r'https://api.tdameritrade.com/v1/accounts'
_HEADERS = {"Content-Type": "application/x-www-form-urlencoded"}

_ACCESS_CODE = utils.retrieve_secret('API_TD_ACCESS_CODE')
_CLIENT_ID = utils.retrieve_secret('API_TD')
_REDIRECT_URI = utils.retrieve_secret('API_TD_REDIRECT_URI')
_REFRESH_TOKEN = utils.retrieve_secret('API_TD_REFRESH_TOKEN')


# Manually refresh API_TD_LOGIN_URL in env
def _get_refresh_token(
        access_code=_ACCESS_CODE,
        auth_url=_AUTH_URL,
        client_id=_CLIENT_ID,
        redirect_uri=_REDIRECT_URI) -> dict:
    """
    Get TD Refresh Token to store in env vars

    1. Update API_TD_ACCESS_CODE in env vars
    2. Run this function
    3. Update API_TD_REFRESH_TOKEN in env vars
    """

    # Url encode access code
    code = urllib.parse.unquote(access_code)

    # Post request
    data = {
        'grant_type': 'authorization_code',
        'access_type': 'offline',
        'code': code,
        'client_id': client_id,
        'redirect_uri': redirect_uri
    }

    # Check if headers are needed
    r = requests.post(url=auth_url, headers=_HEADERS, data=data)
    j = r.json()

    if j.get('error'):
        raise PermissionError(
            'Access token retrieval failed. Maybe refresh: API_TD_ACCESS_CODE')
    else:
        print('Add to env vars as API_TD_REFRESH_TOKEN: ' + j.get('refresh_token'))


def get_access_token(
        refresh_token: str=_REFRESH_TOKEN,
        auth_url: str=_AUTH_URL,
        client_id: str=_CLIENT_ID) -> str:
    """
    Get TD Access Token, valid for 30 minutes
    """

    data = {
        'grant_type': 'refresh_token',
        'refresh_token': refresh_token,
        'client_id': client_id
    }
    r = requests.post(url=auth_url, headers=_HEADERS, data=data)
    j = r.json()
    if j.get('error'):
        raise PermissionError(
            'Access token retrieval failed. Maybe refresh: API_TD_REFRESH_TOKEN')
    access_token = j.get('access_token')
    return access_token


if __name__ == '__main__':
    _get_refresh_token()
