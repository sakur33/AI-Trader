/* ****************************************************************
 * Author: Alejandro
 * Description: 01-create-tables.sql
 * ****************************************************************
 */
CREATE TABLE traders(
    traderid INTEGER PRIMARY KEY AUTOINCREMENT,
    tradername TEXT NOT NULL,
    creation_date TEXT NOT NULL,
    capital REAL NOT NULL,
    max_risk REAL NOT NULL,
    profits REAL NULL,
    losses REAL NULL
);

CREATE TABLE trades(
    tradeid INTEGER PRIMARY KEY AUTOINCREMENT,
    traderid INTEGER NOT NULL,
    date_entry TEXT NOT NULL,
    date_close TEXT NULL,
    symbol TEXT NOT NULL,
    shares INTEGER NOT NULL,
    entry_price REAL NOT NULL,
    entry_stop_loss REAL NOT NULL,
    entry_take_profit REAL NOT NULL,
    out_price REAL NULL,
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