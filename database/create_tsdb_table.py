import psycopg2

dbname = "tick_db"
user = "postgres"
password = "1234"
host = "localhost"
port = 5432
CONNECTION = f"dbname=postgres user={user} password={password} host={host} port={port}"

fd = open("./scripts/V2023.01.05.19.30.15_create-tables_tsdb.sql", "r")
sqlFile = fd.read()
fd.close()

sqlCommands = sqlFile.split(";")

for command in sqlCommands[:-1]:
    # This will skip and report errors
    # For example, if the tables do not yet exist, this will skip over
    # the DROP TABLE commands
    try:
        with psycopg2.connect(CONNECTION) as conn:
            cursor = conn.cursor()
            cursor.execute(command)
            conn.commit()
    except Exception as e:
        print(f"Create tables tsdb error | {e}")
