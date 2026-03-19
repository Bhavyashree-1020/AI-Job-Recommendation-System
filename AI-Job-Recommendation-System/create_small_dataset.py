import pandas as pd

print("Loading dataset...")

df = pd.read_csv("job_descriptions.csv")

print("Selecting important columns...")

df = df[['Job Title','Job Description','skills']]

df = df.dropna()

print("Reducing dataset size...")

df_small = df.sample(20000)

df_small.to_csv("jobs_small.csv", index=False)

print("Small dataset created successfully!")