import os
import pyodbc


if not os.path.exists("./ai_trader.db"):
    with open("./ai_trader.db", "w"):
        pass

try:
    # Trusted Connection to Named Instance
    conn = pyodbc.connect(
        driver="{ODBC Driver 17 for SQL Server}",
        server="localhost",
        port=1433,
        database="master",
        user="sa",
        password="yourStrong(!)Password",
    )
except Exception as e:
    print(f"Exception | create_db_conn | {e}")


fd = open("./scripts/V2023.01.06.21.11.06_create-tables-sql-server.sql", "r")
sqlFile = fd.read()
fd.close()

sqlCommands = sqlFile.split(";")

for command in sqlCommands[:-1]:
    # This will skip and report errors
    # For example, if the tables do not yet exist, this will skip over
    # the DROP TABLE commands
    try:
        cur = conn.cursor()
        cur.execute(command)
        cur.commit()
    except Exception as e:
        print("Command skipped: ", {e})
        print(command)
