import psycopg2

conn = psycopg2.connect(
        host='postgres',
        database='flask_db',
        user='estroev',
        password='estroev'
)

# Open a cursor to perform database operations
cur = conn.cursor()

# Execute a command: this creates a new table
cur.execute('DROP TABLE IF EXISTS users;')
cur.execute(
    'CREATE TABLE users ('
    'id serial PRIMARY KEY,'
    'name varchar (50) NOT NULL,'
    'surname varchar (50) NOT NULL,'
    'username varchar (50) NOT NULL,'
    'pwd_hash varchar (256) NOT NULL,'
    'date_added date DEFAULT CURRENT_TIMESTAMP);'
)
cur.execute(
    'INSERT INTO users (name, surname, username, pwd_hash, date_added) VALUES (%s, %s, %s, %s, %s);',
    ('Egor', 'Stroev', 'estroev', '$2b$12$n9Kfl0QyUmjlXDhP4MIkt.FpWiVEPvMPSHv5ZHYkZpfOll0zl26/e', '2022-06-01'))
conn.commit()
cur.close()
conn.close()