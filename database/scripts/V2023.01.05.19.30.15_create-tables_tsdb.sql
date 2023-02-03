/* ****************************************************************
 * Author: Alejandro
 * Description: create-tables_sqlite
 * ****************************************************************
 */
CREATE TABLE mili_ticks (
    timestamp timestamp NOT NULL,
    symbol text NOT NULL,
    ask double PRECISION NULL,
    bid double PRECISION NULL,
    high double PRECISION NULL,
    low double PRECISION NULL,
    askVolume integer NULL,
    bidVolume integer NULL,
    tick_level integer NULL,
    quoteId integer NULL,
    spreadTable double PRECISION NULL,
    spreadRaw double PRECISION NULL
);
CREATE INDEX ON mili_ticks (symbol, timestamp);
SELECT create_hypertable('mili_ticks', 'timestamp');
CREATE TABLE candles(
    symbol text NOT NULL,
    ctm bigint NOT NULL,
    ctmString timestamp NULL,
    low double PRECISION NULL,
    high double PRECISION NULL,
    open double PRECISION NULL,
    close double PRECISION NULL,
    vol double PRECISION NULL,
    quoteId integer NULL
);
CREATE INDEX ON candles (symbol, ctmString);
SELECT create_hypertable('candles', 'ctmstring');
create table trade_session(
    trader_name text NOT NULL,
    ctmString timestamp NOT NULL,
    symbol text NOT NULL,
    state text not NULL
);
CREATE INDEX ON trade_session (symbol, ctmString);
SELECT create_hypertable('trade_session', 'ctmstring');