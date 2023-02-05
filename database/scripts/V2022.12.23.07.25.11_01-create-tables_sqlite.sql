/* ****************************************************************
 * Author: Alejandro
 * Description: 01-create-tables.sql
 * ****************************************************************
 */
CREATE TABLE traders(
    traderid INTEGER PRIMARY KEY,
    tradername TEXT NOT NULL,
    creation_date TEXT NOT NULL,
    initial_capital REAL NOT NULL,
    max_risk REAL NOT NULL
);

CREATE TABLE trades(
    tradeid INTEGER NOT NULL,
    traderid INTEGER NOT NULL,
    date_entry TEXT NOT NULL,
    date_close TEXT NULL,
    symbol TEXT NOT NULL,
    shares INTEGER NOT NULL,
    entry_price REAL NOT NULL,
    entry_stop_loss REAL NOT NULL,
    entry_take_profit REAL NOT NULL,
    out_price REAL NULL,
    PRIMARY KEY (tradeid, traderid),
    FOREIGN KEY (traderid) REFERENCES traders (traderid)
);

CREATE TABLE balance(
    transactionid INTEGER PRIMARY KEY AUTOINCREMENT,
    traderid INTEGER NOT NULL,
    tradeid INTEGER NOT NULL,
    trans_date TEXT NOT NULL,
    trans_type TEXT NOT NULL,
    balance REAL NOT NULL,
    FOREIGN KEY (traderid) REFERENCES traders (traderid),
    FOREIGN KEY (tradeid) REFERENCES trades (tradeid)
);

CREATE TABLE symbols(
    symbol TEXT NULL,
    min_price REAL NULL,
    currency TEXT NULL,
    categoryName TEXT NULL,
    currencyProfit TEXT NULL,
    quoteId INTEGER NULL,
    quoteIdCross INTEGER NULL,
    marginMode INTEGER NULL,
    profitMode INTEGER NULL,
    pipsPrecision INTEGER NULL,
    contractSize INTEGER NULL,
    exemode INTEGER NULL,
    time INTEGER NULL,
    expiration TEXT NULL,
    stopsLevel INTEGER NULL,
    precision INTEGER NULL,
    swapType INTEGER NULL,
    stepRuleId INTEGER NULL,
    type INTEGER NULL,
    instantMaxVolume INTEGER NULL,
    groupName TEXT NULL,
    description TEXT NULL,
    longOnly INTEGER NULL,
    trailingEnabled INTEGER NULL,
    marginHedgedStrong INTEGER NULL,
    swapEnable INTEGER NULL,
    percentage REAL NULL,
    bid REAL NULL,
    ask REAL NULL,
    high REAL NULL,
    low REAL NULL,
    lotMin REAL NULL,
    lotMax REAL NULL,
    lotStep REAL NULL,
    tickSize REAL NULL,
    tickValue REAL NULL,
    swapLong REAL NULL,
    swapShort REAL NULL,
    leverage REAL NULL,
    spreadRaw REAL NULL,
    spreadTable REAL NULL,
    starting TEXT NULL,
    swap_rollover3days INTEGER NULL,
    marginMaintenance INTEGER NULL,
    marginHedged INTEGER NULL,
    initialMargin INTEGER NULL,
    timeString TEXT NULL,
    shortSelling INTEGER NULL,
    currencyPair INTEGER NULL,
    PRIMARY KEY (symbol, timeString)
);

CREATE TABLE stocks(
    symbol_name TEXT NOT NULL,
    period INT NULL,
    ctm int NOT NULL,
    ctmString TEXT NULL,
    low REAL NULL,
    high REAL NULL,
    open REAL NULL,
    close REAL NULL,
    vol REAL NULL,
    min REAL NULL,
    max REAL NULL,
    PRIMARY KEY (symbol_name, ctm),
    FOREIGN KEY (symbol_name) REFERENCES symbols (symbol)
);

CREATE TABLE ticks (
    timestamp TEXT NOT NULL,
    symbol TEXT NOT NULL,
    ask REAL NULL,
    bid REAL NULL,
    high REAL NULL,
    low REAL NULL,
    askVolume INT NULL,
    bidVolume INT NULL,
    tick_level INT NULL,
    spreadTable REAL NULL,
    spreadRaw REAL NULL
);

CREATE TABLE trading_params(
    symbol_name TEXT NOT NULL,
    date TEXT NOT NULL,
    score REAL NOT NULL,
    short_ma INT NOT NULL,
    long_ma INT NOT NULL,
    out REAL NOT NULL
);