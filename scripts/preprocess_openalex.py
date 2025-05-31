# File: scripts/preprocess_openalex.py

import os
import json
import logging
import argparse
import pandas as pd

# ─────────────────────────────────────────────────────────────────────────────
# 1. ARGPARSE & LOGGING SETUP
# ─────────────────────────────────────────────────────────────────────────────
def parse_args():
    parser = argparse.ArgumentParser(
        description="Preprocess OpenAlex JSON files (arrays or JSONL) into a unified CSV/JSONL."
    )
    parser.add_argument(
        "--input_dir", "-i",
        type=str,
        required=True,
        help="Path to directory containing raw OpenAlex JSON files."
    )
    parser.add_argument(
        "--output_dir", "-o",
        type=str,
        default="data/processed/openalex",
        help="Directory to save processed CSV/JSONL outputs."
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable verbose (DEBUG) logging."
    )
    return parser.parse_args()


def configure_logging(verbose: bool):
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )


# ─────────────────────────────────────────────────────────────────────────────
# 2. FIELD EXTRACTION UTILITIES
# ─────────────────────────────────────────────────────────────────────────────
def extract_authorships(authorships):
    """
    Extract author names from 'authorships' field.
    OpenAlex authorship objects usually look like:
      {"author": {"id": "...", "display_name": "...", …}, …}
    This function returns a comma-separated string of display_names.
    """
    if not isinstance(authorships, list):
        return ""
    names = []
    for a in authorships:
        try:
            if isinstance(a, dict):
                # nested under 'author' key
                author = a.get("author", {})
                name = author.get("display_name", "").strip()
                if name:
                    names.append(name)
        except Exception:
            continue
    return ", ".join(names)


def extract_concepts(concepts):
    """
    Extract concept names from the 'concepts' field.
    Each concept object typically has 'display_name'.
    """
    if not isinstance(concepts, list):
        return ""
    names = []
    for c in concepts:
        try:
            if isinstance(c, dict):
                name = c.get("display_name", "").strip()
                if name:
                    names.append(name)
        except Exception:
            continue
    return ", ".join(names)


def extract_topics(topics):
    """
    Extract topic names (very similar to concepts), but OpenAlex may list 'topics'.
    """
    if not isinstance(topics, list):
        return ""
    names = []
    for t in topics:
        try:
            if isinstance(t, dict):
                name = t.get("display_name", "").strip()
                if name:
                    names.append(name)
        except Exception:
            continue
    return ", ".join(names)


def extract_keywords(keywords):
    """
    OpenAlex 'keywords' field is usually a list of strings.
    """
    if isinstance(keywords, list):
        return ", ".join([str(k).strip() for k in keywords if str(k).strip()])
    return ""


# ─────────────────────────────────────────────────────────────────────────────
# 3. PARSE A SINGLE ARTICLE OBJECT
# ─────────────────────────────────────────────────────────────────────────────
def extract_article_fields(article: dict, source_name: str) -> dict:
    """
    Given one JSON object from OpenAlex, extract a flat dict with these columns:
      - id
      - doi
      - title
      - display_name
      - relevance_score
      - publication_year
      - publication_date
      - language
      - type
      - open_access (boolean)
      - cited_by_count
      - referenced_works_count
      - authors (comma-separated)
      - concepts (comma-separated)
      - topics (comma-separated)
      - keywords (comma-separated)
      - source (filename without .json)
    Returns None if 'id' or 'title' is missing (skip that record).
    """
    try:
        # Required fields: id & title
        art_id = article.get("id", "").strip()
        title = article.get("title", "").strip()
        if not art_id or not title:
            return None

        doi = article.get("doi", "")
        display_name = article.get("display_name", "")
        relevance_score = article.get("relevance_score", "")
        publication_year = article.get("publication_year", "")
        publication_date = article.get("publication_date", "")
        language = article.get("language", "")
        art_type = article.get("type", "")
        open_access = article.get("open_access", False)
        cited_by_count = article.get("cited_by_count", "")
        referenced_works_count = article.get("referenced_works_count", "")

        # Nested fields
        authorships = article.get("authorships", [])
        authors = extract_authorships(authorships)

        concepts = extract_concepts(article.get("concepts", []))
        topics = extract_topics(article.get("topics", []))
        keywords = extract_keywords(article.get("keywords", []))

        return {
            "id": art_id,
            "doi": doi,
            "title": title,
            "display_name": display_name,
            "relevance_score": relevance_score,
            "publication_year": publication_year,
            "publication_date": publication_date,
            "language": language,
            "type": art_type,
            "open_access": open_access,
            "cited_by_count": cited_by_count,
            "referenced_works_count": referenced_works_count,
            "authors": authors,
            "concepts": concepts,
            "topics": topics,
            "keywords": keywords,
            "source": source_name
        }
    except Exception as e:
        logging.debug(f"[extract_article_fields] Skipping invalid article: {e}")
        return None


# ─────────────────────────────────────────────────────────────────────────────
# 4. PROCESS ONE JSON FILE (ARRAY OR JSONL)
# ─────────────────────────────────────────────────────────────────────────────
def process_file(file_path: str, output_records: list):
    """
    Attempt to load `file_path` as a JSON array. If that fails (ValueError), assume
    it's line-delimited JSON (JSONL) and parse line by line. Each valid article
    yields one flat record appended to `output_records`.
    """
    fname = os.path.basename(file_path)
    source_name = os.path.splitext(fname)[0]
    logging.info(f"Processing file: {fname}")

    try:
        # First, try to load as a single JSON array
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        if isinstance(data, list):
            logging.debug(f"  Detected JSON array with {len(data)} articles.")
            for article in data:
                rec = extract_article_fields(article, source_name)
                if rec:
                    output_records.append(rec)
        else:
            logging.warning(f"  Expected a JSON array in {fname}, got {type(data)}. Skipping.")
        return

    except json.JSONDecodeError:
        # If array loading fails, attempt JSONL (one JSON object per line)
        logging.debug(f"  Failed to load as JSON array. Trying JSONL mode for {fname}...")
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                for line_number, line in enumerate(f, start=1):
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        article = json.loads(line)
                        rec = extract_article_fields(article, source_name)
                        if rec:
                            output_records.append(rec)
                    except json.JSONDecodeError as e:
                        logging.debug(f"    Skipping invalid JSON on {fname} line {line_number}: {e}")
            return
        except Exception as e:
            logging.error(f"  Error reading {fname} in JSONL mode: {e}")
            return

    except Exception as e:
        logging.error(f"  Unexpected error while reading {fname}: {e}")
        return


# ─────────────────────────────────────────────────────────────────────────────
# 5. MAIN PREPROCESSING FUNCTION
# ─────────────────────────────────────────────────────────────────────────────
def preprocess_openalex(input_dir: str, output_dir: str):
    logging.info(f"=== Starting OpenAlex preprocessing ===")
    logging.info(f"Raw directory:    {input_dir}")
    logging.info(f"Processed output: {output_dir}")

    if not os.path.isdir(input_dir):
        logging.error(f"Input directory not found: {input_dir}")
        return

    os.makedirs(output_dir, exist_ok=True)

    # Collect all .json files
    files = sorted([f for f in os.listdir(input_dir) if f.lower().endswith(".json")])
    if not files:
        logging.warning("No .json files found in input directory.")
        return

    all_records = []
    for fname in files:
        fpath = os.path.join(input_dir, fname)
        process_file(fpath, all_records)

    if not all_records:
        logging.warning("No valid records extracted from any file.")
        return

    # Build DataFrame
    df = pd.DataFrame(all_records)
    logging.info(f"Total records extracted: {len(df)}")

    # Save to CSV
    csv_path = os.path.join(output_dir, "openalex_processed.csv")
    try:
        df.to_csv(csv_path, index=False)
        logging.info(f"Saved CSV: {csv_path}")
    except Exception as e:
        logging.error(f"Failed to save CSV: {e}")

    # Save to JSONL (one JSON object per line)
    jsonl_path = os.path.join(output_dir, "openalex_processed.json")
    try:
        with open(jsonl_path, "w", encoding="utf-8") as f:
            for rec in all_records:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        logging.info(f"Saved JSONL: {jsonl_path}")
    except Exception as e:
        logging.error(f"Failed to save JSONL: {e}")

    logging.info("=== OpenAlex preprocessing completed successfully ===")


# ─────────────────────────────────────────────────────────────────────────────
# 6. SCRIPT ENTRYPOINT
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    args = parse_args()
    configure_logging(args.verbose)
    preprocess_openalex(args.input_dir, args.output_dir)
