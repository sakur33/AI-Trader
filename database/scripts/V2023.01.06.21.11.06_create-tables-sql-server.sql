/* ****************************************************************
 * Author: Alejandro
 * Description: create-tables-sql-server
 * ****************************************************************
 */
/* ****************************************************************
 * Author: Alejandro
 * Description: 01-create-tables.sql
 * ****************************************************************
 */
CREATE TABLE traders(
    traderid INT IDENTITY (1, 1),
    tradername VARCHAR(50) NOT NULL,
    creation_date DATETIME2 NOT NULL,
    initial_capital DECIMAL(10, 5) NOT NULL,
    max_risk DECIMAL(10, 5) NOT NULL
);
CREATE TABLE trades(
    tradeid INT IDENTITY (1, 1),
    traderid INT NOT NULL,
    symbol VARCHAR(50) NOT NULL,
    time_entry datetime2 NOT NULL,
    time_close datetime2 NULL,
    shares INT NOT NULL,
    entry_price DECIMAL(10, 5) NOT NULL,
    out_price DECIMAL(10, 5) NULL
);
CREATE TABLE balance(
    transactionid INT IDENTITY (1, 1),
    traderid INT NOT NULL,
    tradeid INT NOT NULL,
    transaction_time datetime2 NOT NULL,
    trans_type VARCHAR(50) NOT NULL,
    balance DECIMAL(10, 5) NOT NULL
);
CREATE TABLE symbols(
    symbol VARCHAR(50) NULL,
    min_price DECIMAL(10, 5) NULL,
    currency VARCHAR(50) NULL,
    categoryName VARCHAR(50) NULL,
    currencyProfit VARCHAR(50) NULL,
    quoteId INT NULL,
    quoteIdCross INT NULL,
    marginMode INT NULL,
    profitMode INT NULL,
    pipsPrecision INT NULL,
    contractSize INT NULL,
    exemode INT NULL,
    time INT NULL,
    expiration VARCHAR(50) NULL,
    stopsLevel INT NULL,
    precision INT NULL,
    swapType INT NULL,
    stepRuleId INT NULL,
    type INT NULL,
    instantMaxVolume INT NULL,
    groupName VARCHAR(50) NULL,
    description VARCHAR(50) NULL,
    longOnly INT NULL,
    trailingEnabled INT NULL,
    marginHedgedStrong INT NULL,
    swapEnable INT NULL,
    percentage DECIMAL(10, 5) NULL,
    bid DECIMAL(10, 5) NULL,
    ask DECIMAL(10, 5) NULL,
    high DECIMAL(10, 5) NULL,
    low DECIMAL(10, 5) NULL,
    lotMin DECIMAL(10, 5) NULL,
    lotMax DECIMAL(10, 5) NULL,
    lotStep DECIMAL(10, 5) NULL,
    tickSize DECIMAL(10, 5) NULL,
    tickValue DECIMAL(10, 5) NULL,
    swapLong DECIMAL(10, 5) NULL,
    swapShort DECIMAL(10, 5) NULL,
    leverage DECIMAL(10, 5) NULL,
    spreadRaw DECIMAL(10, 5) NULL,
    spreadTable DECIMAL(10, 5) NULL,
    starting VARCHAR(50) NULL,
    swap_rollover3days INT NULL,
    marginMaintenance INT NULL,
    marginHedged INT NULL,
    initialMargin INT NULL,
    timeString datetime2 NULL,
    shortSelling INT NULL,
    currencyPair INT NULL
);
CREATE TABLE stocks(
    symbol_name VARCHAR(50) NOT NULL,
    period INT NULL,
    ctm int NOT NULL,
    ctmString datetime2 NULL,
    low DECIMAL(10, 5) NULL,
    high DECIMAL(10, 5) NULL,
    [open] DECIMAL(10, 5) NULL,
    [close] DECIMAL(10, 5) NULL,
    vol DECIMAL(10, 5) NULL,
    min DECIMAL(10, 5) NULL,
    max DECIMAL(10, 5) NULL
);
CREATE TABLE trading_params(
    symbol_name VARCHAR(50) NOT NULL,
    date date NOT NULL,
    score DECIMAL(10, 5) NOT NULL,
    short_ma INT NOT NULL,
    long_ma INT NOT NULL,
    [out] DECIMAL(10, 5) NOT NULL
);