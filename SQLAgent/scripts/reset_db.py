import sqlite3, pathlib

db = pathlib.Path(__file__).resolve().parents[1] / "sql_agent_class.db"
seed = pathlib.Path(__file__).resolve().parents[1] / "sql_agent_seed.sql"

print(f"Rebuilding DB at: {db}")
sql = open(seed, "r").read()
conn = sqlite3.connect(db.as_posix())
conn.executescript(sql)
conn.commit()
conn.close()
print("Done.")