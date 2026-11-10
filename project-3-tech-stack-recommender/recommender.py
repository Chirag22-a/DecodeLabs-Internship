"""Recommend technology career paths from a small skills catalog."""

from __future__ import annotations

import csv
import re
import sys
from pathlib import Path

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


CATALOG_FILE = Path(__file__).with_name("raw_skills.csv")
TOP_N = 3
REQUIRED_COLUMNS = {"role", "skills", "tools", "description"}


def load_catalog(path: Path = CATALOG_FILE) -> list[dict[str, str]]:
    """Load and validate the role catalog used by the recommender."""
    try:
        with path.open("r", encoding="utf-8-sig", newline="") as csv_file:
            reader = csv.DictReader(csv_file)
            columns = set(reader.fieldnames or [])
            missing_columns = REQUIRED_COLUMNS - columns
            if missing_columns:
                raise ValueError(
                    "CSV is missing required columns: "
                    + ", ".join(sorted(missing_columns))
                )

            catalog = []
            for line_number, raw_row in enumerate(reader, start=2):
                row = {key: (value or "").strip() for key, value in raw_row.items()}
                if not row["role"] or not row["skills"] or not row["tools"]:
                    raise ValueError(
                        f"CSV row {line_number} needs a role, skills, and tools value."
                    )
                row["description"] = row.get("description", "")
                row["document"] = " ".join(
                    (row["role"], row["skills"], row["tools"], row["description"])
                )
                catalog.append(row)
    except FileNotFoundError as error:
        raise FileNotFoundError(
            f"Could not find {path.name}. Keep it in the same folder as recommender.py."
        ) from error

    if not catalog:
        raise ValueError(f"{path.name} has no role records.")
    return catalog


def parse_interests(user_text: str) -> list[str]:
    """Split comma/semicolon-separated interests and remove duplicates."""
    interests = []
    seen = set()
    for item in re.split(r"[,;]", user_text):
        cleaned = item.strip()
        normalized = cleaned.casefold()
        if cleaned and normalized not in seen:
            interests.append(cleaned)
            seen.add(normalized)
    return interests


def collect_interests() -> list[str]:
    """Ask for at least three skills or career interests."""
    while True:
        answer = input(
            "Enter at least 3 skills or interests, separated by commas\n"
            "Example: Python, Cloud Computing, Automation\n> "
        )
        interests = parse_interests(answer)
        if len(interests) >= 3:
            return interests
        print(f"Please enter at least 3 different interests (you entered {len(interests)}).\n")


def recommend(
    catalog: list[dict[str, str]], interests: list[str], top_n: int = TOP_N
) -> list[dict[str, object]]:
    """Rank catalog roles by TF-IDF cosine similarity to the user profile."""
    vectorizer = TfidfVectorizer(stop_words="english", ngram_range=(1, 2))
    role_vectors = vectorizer.fit_transform(row["document"] for row in catalog)
    user_vector = vectorizer.transform([" ".join(interests)])

    if user_vector.nnz == 0:
        return []

    scores = cosine_similarity(user_vector, role_vectors).ravel()
    ranked_indices = sorted(
        range(len(catalog)),
        key=lambda index: (-scores[index], catalog[index]["role"].casefold()),
    )
    ranked_indices = [index for index in ranked_indices if scores[index] > 0][:top_n]

    feature_names = vectorizer.get_feature_names_out()
    recommendations = []
    for index in ranked_indices:
        contributions = user_vector.multiply(role_vectors[index]).toarray().ravel()
        matched_terms = [
            str(feature_names[feature_index])
            for feature_index in contributions.argsort()[::-1]
            if contributions[feature_index] > 0
        ][:4]
        recommendations.append(
            {
                "role": catalog[index]["role"],
                "skills": catalog[index]["skills"],
                "tools": catalog[index]["tools"],
                "score": float(scores[index]),
                "matched_terms": matched_terms,
            }
        )
    return recommendations


def print_recommendations(
    recommendations: list[dict[str, object]], catalog: list[dict[str, str]]
) -> None:
    if not recommendations:
        known_skills = sorted(
            {
                skill.strip()
                for row in catalog
                for skill in re.split(r"[,;]", row["skills"])
                if skill.strip()
            }
        )
        print("\nNo catalog terms matched those interests. Try skills such as:")
        print(", ".join(known_skills))
        return

    print(f"\nTop {len(recommendations)} career-path recommendations")
    print("=" * 38)
    for rank, item in enumerate(recommendations, start=1):
        print(f"\n{rank}. {item['role']}  |  Similarity: {item['score']:.1%}")
        print(f"   Skills: {item['skills']}")
        print(f"   Suggested tools: {item['tools']}")
        if item["matched_terms"]:
            print(f"   Matching terms: {', '.join(item['matched_terms'])}")

    print("\nSimilarity scores rank this sample catalog; they are not job-market probabilities.")


def main() -> int:
    print("TECH STACK RECOMMENDER")
    print("Find career paths and tools that match your interests.\n")
    try:
        catalog = load_catalog()
        interests = collect_interests()
        print("\nYour interests:", ", ".join(interests))
        print_recommendations(recommend(catalog, interests), catalog)
    except (OSError, ValueError) as error:
        print(f"Error: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
