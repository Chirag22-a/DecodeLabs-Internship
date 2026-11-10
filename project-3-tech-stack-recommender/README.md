# DecodeLabs AI Internship - Project 3

## Tech Stack Recommender

This command-line recommendation system takes at least three skills or career
interests and ranks the three closest career paths in a small technology-role
catalog. For each result it also shows example tools and the terms that matched.

The project turns each role's skills, tools, and description into TF-IDF vectors,
then compares the user's interests to each role with cosine similarity. Results
are sorted by similarity from highest to lowest. The catalog is a curated sample
for this assignment, not a live job listing or a guarantee of job suitability.

## Project files

- `recommender.py` - reads interests, scores roles, and prints the top results.
- `raw_skills.csv` - sample role, skill, tool, and description catalog.
- `requirements.txt` - Python package needed for TF-IDF and cosine similarity.

## Requirements

- Python 3.11 or newer
- The package listed in `requirements.txt`

## Run on Windows

Open PowerShell in this folder and run:

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python recommender.py
```

If PowerShell blocks virtual-environment activation, use:

```powershell
py -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe recommender.py
```

When asked, enter at least three different comma-separated interests. For example:

```text
Python, Cloud Computing, Automation
```

The program ranks the best three matches and prints each role's skills, suggested
tools, cosine similarity score, and matching terms. If none of the entered terms
appear in the catalog, it lists skills you can try. Keep `raw_skills.csv` beside
`recommender.py` when running the program.

## Dataset columns

`raw_skills.csv` contains `role`, `skills`, `tools`, and `description` columns.
The recommendation engine considers all four fields when building role vectors.
