"""
Reads the myScheme dataset (CSV or JSON) and loads it into Postgres as
chunked rows (one row per scheme field), ready for embedding.

Usage:
    python scripts/ingest.py --file data/schemes.json
    python scripts/ingest.py --file data/schemes.csv
"""
import argparse
import json
import os
import sys

import pandas as pd
import psycopg2
from dotenv import load_dotenv

load_dotenv()

POSTGRES_DSN = os.environ["POSTGRES_DSN"]

# Map of (output chunk_type) -> possible source column names in the dataset.
# The HF/Kaggle myScheme dumps don't always use identical headers - check
# your file's columns and adjust this if a field doesn't show up.
FIELD_MAP = {
    "description": ["details", "description", "Description", "scheme_description"],
    "eligibility": ["eligibility", "Eligibility", "eligibility_criteria"],
    "benefits": ["benefits", "Benefits"],
    "application_process": ["application", "application_process", "Application Process", "how_to_apply"],
    "documents": ["documents", "Documents", "documents_required"],
}
NAME_COLS = ["scheme_name", "Scheme Name", "name", "title"]
LINK_COLS = ["official_link", "Official Link", "link", "url"]
# This Kaggle dump has no direct link column - only a slug. Best-guess
# reconstruction of the myScheme detail page URL; spot-check one against
# the live site before trusting it in citations.
SLUG_COLS = ["slug", "Slug"]


def first_present(row, candidates):
    for c in candidates:
        if c in row and pd.notna(row[c]) and str(row[c]).strip():
            return str(row[c]).strip()
    return None


def load_dataframe(path: str) -> pd.DataFrame:
    if path.endswith(".json"):
        with open(path) as f:
            data = json.load(f)
        return pd.DataFrame(data)
    return pd.read_csv(path)


def main(path: str):
    df = load_dataframe(path)
    print(f"Loaded {len(df)} scheme rows from {path}")

    conn = psycopg2.connect(POSTGRES_DSN)
    cur = conn.cursor()

    inserted = 0
    for _, row in df.iterrows():
        scheme_name = first_present(row, NAME_COLS)
        official_link = first_present(row, LINK_COLS)
        if not official_link:
            slug = first_present(row, SLUG_COLS)
            if slug:
                official_link = f"https://www.myscheme.gov.in/schemes/{slug}"
        if not scheme_name:
            continue

        for chunk_type, cols in FIELD_MAP.items():
            content = first_present(row, cols)
            if not content:
                continue
            # Prepend the scheme name so it's actually present in the text
            # that gets full-text indexed and embedded - many scheme
            # descriptions never restate their own name in the body text,
            # which otherwise makes the single most identifying word in a
            # user's question invisible to search.
            indexed_content = f"{scheme_name}. {content}"
            cur.execute(
                """
                INSERT INTO scheme_chunks (scheme_name, chunk_type, content, official_link)
                VALUES (%s, %s, %s, %s)
                """,
                (scheme_name, chunk_type, indexed_content, official_link),
            )
            inserted += 1

    conn.commit()
    cur.close()
    conn.close()
    print(f"Inserted {inserted} chunks into scheme_chunks")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--file", required=True, help="Path to schemes.json or schemes.csv")
    args = parser.parse_args()
    if not os.path.exists(args.file):
        sys.exit(f"File not found: {args.file}")
    main(args.file)