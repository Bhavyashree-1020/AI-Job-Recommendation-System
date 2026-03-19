import sqlite3
import pandas as pd

print("Starting database setup...")

# Connect to SQLite database
conn = sqlite3.connect("jobs.db")
cursor = conn.cursor()

# Create users table
cursor.execute("""
CREATE TABLE IF NOT EXISTS users (
    user_id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT,
    email TEXT,
    skills TEXT,
    qualification TEXT,
    experience TEXT
)
""")

# Create jobs table
cursor.execute("""
CREATE TABLE IF NOT EXISTS jobs (
    job_id INTEGER PRIMARY KEY,
    job_title TEXT,
    company TEXT,
    location TEXT,
    skills TEXT,
    job_description TEXT,
    salary_range TEXT
)
""")

# Create recommendations table
cursor.execute("""
CREATE TABLE IF NOT EXISTS recommendations (
    rec_id INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id INTEGER,
    job_id INTEGER,
    similarity_score REAL,
    FOREIGN KEY (user_id) REFERENCES users(user_id),
    FOREIGN KEY (job_id) REFERENCES jobs(job_id)
)
""")

print("Tables created successfully!")

# Load original dataset (only first 1000 rows for speed)
df = pd.read_csv("job_descriptions.csv", nrows=1000)

print("Dataset loaded successfully!")
print("Columns in dataset:")
print(df.columns)

# Insert jobs into jobs table
for _, row in df.iterrows():
    cursor.execute("""
    INSERT OR IGNORE INTO jobs (job_id, job_title, company, location, skills, job_description, salary_range)
    VALUES (?, ?, ?, ?, ?, ?, ?)
    """, (
        int(row["Job Id"]),
        str(row["Job Title"]),
        str(row["Company"]),
        str(row["location"]),
        str(row["skills"]),
        str(row["Job Description"]),
        str(row["Salary Range"])
    ))

conn.commit()
conn.close()

print("Database and tables created successfully!")
print("Jobs inserted into database!")