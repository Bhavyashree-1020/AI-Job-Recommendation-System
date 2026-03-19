import streamlit as st
import pandas as pd
import sqlite3
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

st.set_page_config(page_title="AI Job Recommendation System", layout="wide")

st.title("AI-Based Job Recommendation System")
st.write("A smart system using Data Analytics, Machine Learning, and DBMS")

# -------------------------------
# DATABASE CONNECTION
# -------------------------------
conn = sqlite3.connect("jobs.db", check_same_thread=False)
cursor = conn.cursor()

# -------------------------------
# LOAD DATASET (USE ORIGINAL FILE)
# -------------------------------
df = pd.read_csv("job_descriptions.csv", nrows=1000)

# -------------------------------
# DATA CLEANING (DADV)
# -------------------------------
df.drop_duplicates(inplace=True)
df.dropna(subset=["Job Title", "Job Description", "skills"], inplace=True)

df["skills"] = df["skills"].astype(str).str.lower()
df["Job Description"] = df["Job Description"].astype(str).str.lower()

# Fill missing optional columns
df["Company"] = df["Company"].fillna("Not Available")
df["location"] = df["location"].fillna("Not Available")

# -------------------------------
# PREPROCESSING
# -------------------------------
df["text"] = df["Job Description"] + " " + df["skills"]

# -------------------------------
# TF-IDF VECTORIZATION
# -------------------------------
vectorizer = TfidfVectorizer(stop_words="english")
tfidf_matrix = vectorizer.fit_transform(df["text"])

# -------------------------------
# USER INPUT SECTION
# -------------------------------
st.subheader("Enter Your Details")

name = st.text_input("Enter your name")
email = st.text_input("Enter your email")
qualification = st.text_input("Enter your qualification")
experience = st.text_input("Enter your experience")
user_skill = st.text_input("Enter your skills (example: python, machine learning, data analysis)")

# -------------------------------
# RECOMMENDATION BUTTON
# -------------------------------
if st.button("Recommend Jobs"):

    if user_skill.strip() == "" or name.strip() == "" or email.strip() == "":
        st.warning("Please enter at least Name, Email, and Skills.")
    else:
        # Save user to DB
        cursor.execute("""
            INSERT INTO users (name, email, skills, qualification, experience)
            VALUES (?, ?, ?, ?, ?)
        """, (name, email, user_skill, qualification, experience))
        conn.commit()

        user_id = cursor.lastrowid

        # Convert user input to vector
        user_vector = vectorizer.transform([user_skill.lower()])

        # Calculate similarity
        similarity = cosine_similarity(user_vector, tfidf_matrix)

        # Get top 10 matching jobs
        top_indices = similarity[0].argsort()[-10:][::-1]

        st.subheader("Top Job Recommendations")

        jobs_shown = []
        shown_count = 0

        for i in top_indices:
            title = df.iloc[i]["Job Title"]
            company = df.iloc[i]["Company"]
            location = df.iloc[i]["location"]
            score = round(similarity[0][i] * 100, 2)

            if title not in jobs_shown and shown_count < 5:
                jobs_shown.append(title)

                st.write(f"**Job Title:** {title}")
                st.write(f"**Company:** {company}")
                st.write(f"**Location:** {location}")
                st.write(f"**Match Score:** {score}%")
                st.write("---")

                # Save recommendation in DB
                job_id = int(df.iloc[i]["Job Id"])

                cursor.execute("""
                    INSERT INTO recommendations (user_id, job_id, similarity_score)
                    VALUES (?, ?, ?)
                """, (user_id, job_id, score))
                conn.commit()

                shown_count += 1

# -------------------------------
# DATA VISUALIZATION (DADV)
# -------------------------------
st.subheader("Top 10 Most Common Skills in Dataset")

skill_counts = (
    df["skills"]
    .str.split(",")
    .explode()
    .str.strip()
    .value_counts()
    .head(10)
)

st.bar_chart(skill_counts)