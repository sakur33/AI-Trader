import os
import sqlite3


if not os.path.exists("./ai_trader.db"):
    with open("./ai_trader.db", "w"):
        pass

try:
    conn = sqlite3.connect(f"./ai_trader.db")
except Exception as e:
    print(f"Exception | create_db_conn | {e}")


fd = open("./scripts/V2022.12.23.07.25.11_01-create-tables.sql", "r")
sqlFile = fd.read()
fd.close()

sqlCommands = sqlFile.split(";")

for command in sqlCommands:
    # This will skip and report errors
    # For example, if the tables do not yet exist, this will skip over
    # the DROP TABLE commands
    try:
        cur = conn.cursor()
        cur.execute(command)
    except Exception as e:
        print("Command skipped: ", {e})
