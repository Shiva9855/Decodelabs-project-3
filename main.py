"""
AI Recommendation Logic — Career/Job Role Recommender
=======================================================

Project 3: AI Recommendation Logic

Goal:
    A simple recommendation system that takes a user's skills/interests
    as input and recommends the job roles that match best, using
    logic-based pattern matching (set overlap / similarity scoring).

    This version loads the dataset from "raw_skills.csv" using pandas
    (df = pd.read_csv("raw_skills.csv")). Keep the CSV file in the
    SAME folder as this script.

How it works (the "recommendation logic"):
    1. Load raw_skills.csv into a pandas DataFrame.
    2. Ask the user to type in the skills/interests they have.
    3. Turn the user's skills and each job's skills into sets, and
       measure how well they overlap using Jaccard Similarity:

            similarity = |A ∩ B| / |A ∪ B|

       where A = user's skill set, B = a job's required skill set.
       This is a classic "pattern matching" technique: the more skills
       in common (relative to the total distinct skills involved),
       the higher the match score.
    4. Rank every job by its similarity score and display the
       top matches, along with which skills matched and which
       skills the user could learn next to be an even stronger fit.

Requirements:
    pip install pandas

Run it:
    python3 recommender.py
    (make sure raw_skills.csv is in the same folder)
"""

import os
import sys

import pandas as pd


CSV_FILE = "raw_skills.csv"


# ---------------------------------------------------------------------------
# Data loading (pandas)
# ---------------------------------------------------------------------------

def load_dataset(path):
    """Read the CSV file into a DataFrame and clean it up a bit."""
    # Try utf-8 first; fall back to cp1252/latin-1 if the file has
    # Windows-style special characters (curly quotes, en-dashes, etc.)
    # that raise UnicodeDecodeError under strict utf-8.
    try:
        df = pd.read_csv(path, encoding="utf-8")
    except UnicodeDecodeError:
        try:
            df = pd.read_csv(path, encoding="cp1252")
        except UnicodeDecodeError:
            df = pd.read_csv(path, encoding="latin-1")

    # make sure the expected columns exist
    required_cols = {"Job Title", "Job Description", "Skills", "Certifications"}
    missing_cols = required_cols - set(df.columns)
    if missing_cols:
        raise ValueError(f"CSV is missing expected column(s): {missing_cols}")

    # drop rows with no title or no skills
    df = df.dropna(subset=["Job Title", "Skills"]).reset_index(drop=True)

    # turn the comma-separated "Skills" and "Certifications" strings into lists
    df["SkillsList"] = df["Skills"].apply(
        lambda s: [x.strip() for x in str(s).split(",") if x.strip()]
    )
    df["CertsList"] = df["Certifications"].apply(
        lambda c: [x.strip() for x in str(c).split(",") if x.strip()] if pd.notna(c) else []
    )

    return df


# ---------------------------------------------------------------------------
# Matching logic (the "AI recommendation logic")
# ---------------------------------------------------------------------------

def normalize(skill):
    """Lowercase + strip so 'Python' and 'python ' are treated as the same skill."""
    return skill.strip().lower()


def jaccard_similarity(set_a, set_b):
    """Return overlap between two sets as a 0.0-1.0 score."""
    if not set_a or not set_b:
        return 0.0
    intersection = set_a & set_b
    union = set_a | set_b
    return len(intersection) / len(union)


def recommend(user_skills, df, top_n=5):
    """
    Score every job (row of the DataFrame) against the user's skills
    and return the top_n best matches, sorted from best to worst.

    Returns a list of dicts, each with the job info + score + matched/missing skills.
    """
    user_set = {normalize(s) for s in user_skills}

    results = []
    for _, row in df.iterrows():
        job_skills = row["SkillsList"]
        job_set = {normalize(s) for s in job_skills}

        score = jaccard_similarity(user_set, job_set)
        matched = [s for s in job_skills if normalize(s) in user_set]
        missing = [s for s in job_skills if normalize(s) not in user_set]

        results.append({
            "title": row["Job Title"],
            "description": row["Job Description"] if pd.notna(row["Job Description"]) else "",
            "certifications": row["CertsList"],
            "score": score,
            "matched": matched,
            "missing": missing,
        })

    # sort by score, descending
    results.sort(key=lambda r: r["score"], reverse=True)
    return results[:top_n]


# ---------------------------------------------------------------------------
# Display helpers
# ---------------------------------------------------------------------------

def match_bar(score, width=20):
    """A little ASCII progress bar, e.g. [##########----------] 50%"""
    filled = int(round(score * width))
    bar = "#" * filled + "-" * (width - filled)
    return f"[{bar}] {score * 100:5.1f}%"


def print_recommendations(results):
    if not results or all(r["score"] == 0 for r in results):
        print("\nNo matching roles found. Try adding a few more skills!\n")
        return

    print("\n" + "=" * 60)
    print("TOP RECOMMENDATIONS")
    print("=" * 60)

    for rank, r in enumerate(results, start=1):
        print(f"\n#{rank}  {r['title']}")
        print(f"     {match_bar(r['score'])}")
        if r["description"]:
            print(f"     What it involves: {r['description']}")
        if r["matched"]:
            print(f"     [+] Skills you already have: {', '.join(r['matched'])}")
        if r["missing"]:
            preview = r["missing"][:6]
            more = f" (+{len(r['missing']) - 6} more)" if len(r["missing"]) > 6 else ""
            print(f"     [>] Skills to learn next: {', '.join(preview)}{more}")
        if r["certifications"]:
            print(f"     [*] Related certifications: {', '.join(r['certifications'][:2])}")

    print("\n" + "=" * 60)


# ---------------------------------------------------------------------------
# Interactive program
# ---------------------------------------------------------------------------

def ask_for_skills():
    print("Enter your skills or interests, separated by commas.")
    print("Example: Python, Machine Learning, SQL, Cloud Computing\n")
    raw = input("Your skills: ").strip()
    return [s for s in raw.split(",") if s.strip()]


def ask_for_top_n(default=5):
    raw = input(f"How many recommendations would you like to see? [{default}]: ").strip()
    if not raw:
        return default
    try:
        n = int(raw)
        return max(1, n)
    except ValueError:
        return default


def main():
    if not os.path.exists(CSV_FILE):
        print(f"'{CSV_FILE}' not found. Put it in the same folder as this script.")
        sys.exit(1)

    df = load_dataset(CSV_FILE)
    print(f"Loaded {len(df)} job roles from '{CSV_FILE}'.\n")

    while True:
        user_skills = ask_for_skills()
        if not user_skills:
            print("Please enter at least one skill.\n")
            continue

        top_n = ask_for_top_n()
        results = recommend(user_skills, df, top_n=top_n)
        print_recommendations(results)

        again = input("\nTry another search? (y/n): ").strip().lower()
        if again != "y":
            print("\nThanks for using the Career Recommender. Good luck!")
            break
        print()


if __name__ == "__main__":
    main()