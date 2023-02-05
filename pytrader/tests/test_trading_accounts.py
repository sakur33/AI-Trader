from pytrader.src.custom_exception import LoginError
from pytrader.src.trading_accounts import Trader


def test_trader_bad_credentials() -> None:
    try:
        trader = Trader(user="", passwd="")
    except Exception as e:
        assert isinstance(e, LoginError)
