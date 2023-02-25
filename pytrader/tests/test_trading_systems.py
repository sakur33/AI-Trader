from pytrader.src import custom_exceptions
from pytrader.src import trading_systems


def test_trader_bad_credentials() -> None:
    try:
        trader = Trader(user="", passwd="")
    except Exception as e:
        assert isinstance(e, LoginError)
