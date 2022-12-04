from tqdm import tqdm
from API import XTB
from creds import user, passw
from utils import candles_to_excel
import time

API = XTB(user, passw)
symbols = API.get_AllSymbols()
STCs = API.get_STCSymbols()
candles = False
for STC in tqdm(STCs):
    candles = API.get_Candles(period="M1", symbol=STC['symbol'])
    time.sleep(5)
    candles_to_excel(candles, "../data", str(STC['symbol']) + ".xlsx", API.exec_start)
API.logout()