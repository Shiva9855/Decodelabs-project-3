# Decodelabs-project-3
Project 3: AI Recommendation Logic Goal: Create a simple recommendation system based on user preferences. Key Requirements: • Take user input (choices or interests) • Match preferences using logic or similarity • Display recommended items Key Skills: Logic building, pattern matching, recommendation concepts 


# Project 3 — AI Recommendation Logic
### Career/Job Role Recommender (pandas + CSV file)

A simple recommendation system that suggests job roles based on the
skills or interests a user enters, matched using logic-based pattern
matching. This version loads the dataset from **raw_skills.csv** using
pandas — `df = pd.read_csv("raw_skills.csv")`.

## Files
- `main.py` — the program (run this)
- `raw_skills.csv` — dataset of 493 job roles (Job Title, Job Description, Skills, Certifications)

**Keep both files in the same folder.**

## Setup
```
pip install pandas
```

## How to run
```
python3 recommender.py
```
You'll be prompted to:
1. Enter your skills/interests, comma-separated (e.g. `Python, SQL, Machine Learning`)
2. Choose how many recommendations to display
3. View ranked results with a match percentage, skills you already have, and skills to learn next

## How the matching logic works
Each job role has a list of required skills. The user's entered skills
and each job's skills are converted into sets, and compared using
**Jaccard Similarity**:

```
similarity = |A ∩ B| / |A ∪ B|
```

- `A` = the user's skill set
- `B` = a job's required skill set
- `A ∩ B` = skills in common
- `A ∪ B` = all distinct skills across both

The higher the overlap (relative to all distinct skills involved), the
higher the match score. Jobs are sorted highest to lowest, and results
show:
- **Matched skills** — what the user already has
- **Skills to learn next** — what's missing for that role
- **Related certifications** — from the dataset

## Possible extensions
- Weight rarer/specialized skills higher (TF-IDF style)
- Add a simple GUI or web front-end
- Let users rate recommendations to refine future results
- Combine skill matching with keyword matching against job descriptions
