"""Extract raw Yelp Open Dataset JSON files into typed parquet.

One-to-one extraction — no filtering, no feature engineering, no data loss.
Produces a parquet file for each source JSON that faithfully represents
the full dataset with correct dtypes.

Usage:
    python scripts/extract_yelp.py --raw-dir data/raw --out-dir data/extracted

Expected raw files in raw-dir/:
    - yelp_academic_dataset_business.json
    - yelp_academic_dataset_review.json
    - yelp_academic_dataset_user.json
    - yelp_academic_dataset_checkin.json
    - yelp_academic_dataset_tip.json

Outputs to out-dir/:
    - business.parquet
    - review.parquet
    - user.parquet
    - checkin.parquet   (flattened: one row per business × timestamp)
    - tip.parquet
"""

import argparse
import json
from pathlib import Path

import pandas as pd


# ── Loaders ──────────────────────────────────────────────────────────────────
# Each loader reads the raw JSON and returns a DataFrame with correct dtypes.
# No rows are dropped. No columns are derived. This is pure extraction.


def load_businesses(filepath: Path) -> pd.DataFrame:
    """Load business.json → DataFrame.

    Columns:
        business_id (str), name (str), address (str), city (str),
        state (str), postal_code (str), latitude (float64),
        longitude (float64), stars (float64), review_count (int64),
        is_open (int64), attributes (object/dict), categories (object/list),
        hours (object/dict)
    """
    records = []
    with open(filepath, "r") as f:
        for line in f:
            records.append(json.loads(line))

    df = pd.DataFrame(records)

    # Ensure correct dtypes for known columns
    dtype_map = {
        "business_id": "string",
        "name": "string",
        "address": "string",
        "city": "string",
        "state": "string",
        "postal_code": "string",
        "latitude": "float64",
        "longitude": "float64",
        "stars": "float64",
        "review_count": "int64",
        "is_open": "int64",
    }
    for col, dtype in dtype_map.items():
        if col in df.columns:
            df[col] = df[col].astype(dtype)

    # categories: the docs show a JSON array, but some dataset versions
    # return a comma-separated string. Normalize to list[str] either way.
    if "categories" in df.columns:
        df["categories"] = df["categories"].apply(_normalize_categories)

    print(f"  business: {len(df):,} rows, {len(df.columns)} columns")
    return df


def load_reviews(filepath: Path) -> pd.DataFrame:
    """Load review.json → DataFrame.

    Columns:
        review_id (str), user_id (str), business_id (str), stars (int64),
        date (datetime64), text (str), useful (int64), funny (int64),
        cool (int64)

    Note: Yelp review dates are YYYY-MM-DD with no time-of-day component.
    """
    records = []
    with open(filepath, "r") as f:
        for line in f:
            records.append(json.loads(line))

    df = pd.DataFrame(records)

    dtype_map = {
        "review_id": "string",
        "user_id": "string",
        "business_id": "string",
        "stars": "int64",
        "useful": "int64",
        "funny": "int64",
        "cool": "int64",
        "text": "string",
    }
    for col, dtype in dtype_map.items():
        if col in df.columns:
            df[col] = df[col].astype(dtype)

    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"])

    print(f"  review: {len(df):,} rows, {len(df.columns)} columns")
    return df


def load_users(filepath: Path) -> pd.DataFrame:
    """Load user.json → DataFrame.

    Columns:
        user_id (str), name (str), review_count (int64),
        yelping_since (datetime64), friends (object/list),
        useful (int64), funny (int64), cool (int64), fans (int64),
        elite (object/list), average_stars (float64),
        compliment_* (int64 each)
    """
    records = []
    with open(filepath, "r") as f:
        for line in f:
            records.append(json.loads(line))

    df = pd.DataFrame(records)

    dtype_map = {
        "user_id": "string",
        "name": "string",
        "review_count": "int64",
        "useful": "int64",
        "funny": "int64",
        "cool": "int64",
        "fans": "int64",
        "average_stars": "float64",
    }
    for col, dtype in dtype_map.items():
        if col in df.columns:
            df[col] = df[col].astype(dtype)

    # Cast all compliment columns to int64
    compliment_cols = [c for c in df.columns if c.startswith("compliment_")]
    for col in compliment_cols:
        df[col] = df[col].astype("int64")

    if "yelping_since" in df.columns:
        df["yelping_since"] = pd.to_datetime(df["yelping_since"])

    # friends: docs show array of strings, but some versions give
    # comma-separated string. Normalize to list[str].
    if "friends" in df.columns:
        df["friends"] = df["friends"].apply(_normalize_string_list)

    # elite: docs show array of ints, but some versions give
    # comma-separated string. Normalize to list[int].
    if "elite" in df.columns:
        df["elite"] = df["elite"].apply(_normalize_elite)

    print(f"  user: {len(df):,} rows, {len(df.columns)} columns")
    return df


def load_checkins(filepath: Path) -> pd.DataFrame:
    """Load checkin.json → DataFrame, flattened to one row per timestamp.

    Raw format is one row per business with a comma-separated date string.
    We flatten so each (business_id, timestamp) is its own row, which makes
    aggregation straightforward downstream.

    Columns:
        business_id (str), timestamp (datetime64)
    """
    rows = []
    with open(filepath, "r") as f:
        for line in f:
            obj = json.loads(line)
            biz_id = obj["business_id"]
            date_str = obj.get("date", "")
            if date_str:
                for ts in date_str.split(", "):
                    ts = ts.strip()
                    if ts:
                        rows.append({"business_id": biz_id, "timestamp": ts})

    df = pd.DataFrame(rows)

    if len(df) > 0:
        df["business_id"] = df["business_id"].astype("string")
        df["timestamp"] = pd.to_datetime(df["timestamp"])

    print(f"  checkin: {len(df):,} rows (flattened from per-business format)")
    return df


def load_tips(filepath: Path) -> pd.DataFrame:
    """Load tip.json → DataFrame.

    Columns:
        user_id (str), business_id (str), text (str),
        date (datetime64), compliment_count (int64)
    """
    records = []
    with open(filepath, "r") as f:
        for line in f:
            records.append(json.loads(line))

    df = pd.DataFrame(records)

    dtype_map = {
        "user_id": "string",
        "business_id": "string",
        "text": "string",
        "compliment_count": "int64",
    }
    for col, dtype in dtype_map.items():
        if col in df.columns:
            df[col] = df[col].astype(dtype)

    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"])

    print(f"  tip: {len(df):,} rows, {len(df.columns)} columns")
    return df


# ── Helpers ──────────────────────────────────────────────────────────────────


def _normalize_categories(val) -> list[str] | None:
    """Normalize categories to list[str] regardless of source format."""
    if val is None:
        return None
    if isinstance(val, list):
        return [str(c).strip() for c in val if c]
    if isinstance(val, str):
        return [c.strip() for c in val.split(",") if c.strip()]
    return None


def _normalize_string_list(val) -> list[str]:
    """Normalize a field that may be a list or comma-separated string."""
    if val is None:
        return []
    if isinstance(val, list):
        return [str(x).strip() for x in val if x]
    if isinstance(val, str):
        s = val.strip()
        if not s or s.lower() == "none":
            return []
        return [x.strip() for x in s.split(",") if x.strip()]
    return []


def _normalize_elite(val) -> list[int]:
    """Normalize elite years to list[int]."""
    if val is None:
        return []
    if isinstance(val, list):
        return [int(x) for x in val if x]
    if isinstance(val, str):
        s = val.strip()
        if not s or s.lower() == "none":
            return []
        parts = []
        for x in s.split(","):
            x = x.strip()
            if x.isdigit():
                parts.append(int(x))
        return parts
    return []


# ── Main ─────────────────────────────────────────────────────────────────────


SOURCE_FILES = {
    "business": ("yelp_academic_dataset_business.json", load_businesses),
    "review": ("yelp_academic_dataset_review.json", load_reviews),
    "user": ("yelp_academic_dataset_user.json", load_users),
    "checkin": ("yelp_academic_dataset_checkin.json", load_checkins),
    "tip": ("yelp_academic_dataset_tip.json", load_tips),
}


def main():
    parser = argparse.ArgumentParser(
        description="Extract Yelp Open Dataset JSON → parquet (no filtering)"
    )
    parser.add_argument(
        "--raw-dir", default="data/raw",
        help="Directory containing raw Yelp JSON files"
    )
    parser.add_argument(
        "--out-dir", default="data/extracted",
        help="Output directory for parquet files"
    )
    parser.add_argument(
        "--only", nargs="*", choices=list(SOURCE_FILES.keys()),
        help="Extract only specific files (default: all)"
    )
    args = parser.parse_args()

    raw_dir = Path(args.raw_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    targets = args.only if args.only else list(SOURCE_FILES.keys())

    print(f"Extracting from: {raw_dir}")
    print(f"Output to: {out_dir}\n")

    for name in targets:
        filename, loader = SOURCE_FILES[name]
        filepath = raw_dir / filename

        if not filepath.exists():
            print(f"  SKIP {name}: {filepath} not found")
            continue

        print(f"Extracting {name}...")
        df = loader(filepath)
        out_path = out_dir / f"{name}.parquet"
        df.to_parquet(out_path, index=False)
        print(f"  → {out_path} ({out_path.stat().st_size / 1024 / 1024:.1f} MB)\n")

    print("Extraction complete.")


if __name__ == "__main__":
    main()