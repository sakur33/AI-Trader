import psycopg2

dbname = "tick_db"
user = "postgres"
password = "1234"
host = "localhost"
port = 5432
CONNECTION = f"dbname =tsdb user={user} password={password} host={host}port={port} sslmode=require"

with psycopg2.connect(CONNECTION) as conn:
    cursor = conn.cursor()
    cursor.execute(
        "CREATE TABLE ticks (timestamp timestamptz NOT NULL,symbol varchar(50) NOT NULL,ask double PRECISION NULL,bid double PRECISION NULL,high double PRECISION NULL,low double PRECISION NULL,askVolume integer NULL,bidVolume integer NULL,tick_level integer NULL,quoteId integer NULL,spreadTable double PRECISION NULL,spreadRaw double PRECISION NULL);"
    )
    cursor.commit()
