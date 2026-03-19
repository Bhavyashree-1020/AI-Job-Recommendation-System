import sqlite3

# Connect to database
conn = sqlite3.connect("jobs.db")
cursor = conn.cursor()

# Count total jobs
cursor.execute("SELECT COUNT(*) FROM jobs")
count = cursor.fetchone()[0]

print(f"Total jobs in database: {count}")

# Show first 5 jobs
cursor.execute("SELECT job_id, job_title, company, location FROM jobs LIMIT 5")
rows = cursor.fetchall()

print("\nFirst 5 jobs in database:")
for row in rows:
    print(row)

conn.close()