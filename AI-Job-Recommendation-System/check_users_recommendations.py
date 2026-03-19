import sqlite3

# Connect to database
conn = sqlite3.connect("jobs.db")
cursor = conn.cursor()

# Check users table
cursor.execute("SELECT * FROM users")
users = cursor.fetchall()

print("USERS TABLE:")
for user in users:
    print(user)

print("\n" + "="*50 + "\n")

# Check recommendations table
cursor.execute("SELECT * FROM recommendations")
recs = cursor.fetchall()

print("RECOMMENDATIONS TABLE:")
for rec in recs:
    print(rec)

conn.close()