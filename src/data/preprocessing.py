"""Preprocessing pipeline for extracted Yelp Open Dataset parquets.

Handles:
- Filtering to restaurant/food categories
- Extracting and encoding features for both towers
- Aggregating checkin data into business-level temporal profiles
- Review and user filtering
- Pre-COVID date cutoff for pandemic-distorted data

Reads from: data/extracted/ (output of extract_yelp.py)
Writes to:  data/processed/

Owners: Antonio (user side), Danny (restaurant side)
"""

import math
from zoneinfo import ZoneInfo

import pandas as pd
import numpy as np


# IANA timezone mapping for Yelp Open Dataset states/provinces
STATE_TO_TZ = {
    "PA": "America/New_York",    "FL": "America/New_York",
    "NJ": "America/New_York",    "DE": "America/New_York",
    "IN": "America/New_York",    "MA": "America/New_York",
    "TN": "America/Chicago",     "MO": "America/Chicago",
    "LA": "America/Chicago",     "IL": "America/Chicago",
    "TX": "America/Chicago",
    "AZ": "America/Phoenix",     # no DST
    "CO": "America/Denver",      "ID": "America/Boise",
    "UT": "America/Denver",
    "NV": "America/Los_Angeles", "CA": "America/Los_Angeles",
    "WA": "America/Los_Angeles",
    "AB": "America/Edmonton",    "HI": "Pacific/Honolulu",
    "MT": "America/Denver",
    "NC": "America/New_York",
    "XMS": "America/Chicago",     # Tamaulipas, Mexico (Yelp convention)
}

DEFAULT_TZ = "America/New_York"  # fallback for unmapped states

def localize_to_local(
    df: pd.DataFrame, ts_col: str, tz_col: str
) -> pd.Series:
    """Convert UTC timestamps to local time, vectorized by timezone group.

    Groups by timezone to avoid row-by-row apply(). Since the Yelp dataset
    only spans ~6 timezones, this is effectively O(n) with tiny constant.

    Args:
        df: DataFrame containing timestamps and timezone names.
        ts_col: Column name of UTC timestamps (datetime64).
        tz_col: Column name of IANA timezone strings.

    Returns:
        Series of naive datetimes in local time (tzinfo stripped so
        mixed-tz groups concat without dtype conflicts).
    """
    pieces = []
    for tz_name, group in df.groupby(tz_col):
        local = (
            group[ts_col]
            .dt.tz_localize("UTC")
            .dt.tz_convert(str(tz_name))
            .dt.tz_localize(None)
        )
        pieces.append(local)
    return pd.concat(pieces).sort_index()


def cyclical_encode(value: float, period: float) -> tuple[float, float]:
    """Encode a cyclic value as (sin, cos) pair.

    Args:
        value: The value to encode (e.g., hour 0-23, day 0-6).
        period: The period of the cycle (e.g., 24 for hours, 7 for days).

    Returns:
        Tuple of (sin, cos) floats.
    """
    angle = 2 * math.pi * value / period
    return math.sin(angle), math.cos(angle)


def _apply_date_cutoff(
    df: pd.DataFrame, date_col: str, date_cutoff: str | None
) -> pd.DataFrame:
    """Filter rows before a date cutoff. No-op if cutoff is None."""
    if date_cutoff is None:
        return df
    return df[df[date_col] < pd.Timestamp(date_cutoff)]


# ── Business preprocessing ───────────────────────────────────────────────────


def is_restaurant(categories):
    if categories is None:
        return False
    if isinstance(categories, str):
        categories = [c.strip() for c in categories.split(",")]
    cats_lower = {c.strip().lower() for c in categories}
    return "restaurants" in cats_lower


def preprocess_businesses(df: pd.DataFrame) -> pd.DataFrame:
    """Filter to restaurants and extract structured features.

    Expects extracted parquet where categories is an array-like of strings
    per row (numpy array from parquet, or list).

    Returns DataFrame with all original columns plus price_tier,
    filtered to restaurants only.
    """
    sample = df["categories"].iloc[0]
    assert not isinstance(sample, str), (
        f"categories must be array-like of strings, got raw str. "
        f"Check extract_yelp.py — split on commas before saving."
    )
    assert hasattr(sample, "__iter__") and not isinstance(sample, str), (
        f"categories must be array-like of strings (list or ndarray), "
        f"got {type(sample).__name__}. Check extract_yelp.py."
    )
    
    mask = df["categories"].apply(is_restaurant)
    restaurants = df[mask].copy()

    # Extract price tier from attributes (RestaurantsPriceRange2)
    restaurants["price_tier"] = (
        restaurants["attributes"]
        .apply(lambda x: x.get("RestaurantsPriceRange2") if isinstance(x, dict) else None)
        .apply(lambda x: int(x) if x and str(x).isdigit() else 2)  # default to mid-range
    )

    # Map state to IANA timezone for local-time conversion downstream
    restaurants["tz"] = restaurants["state"].map(STATE_TO_TZ)
    unmapped = restaurants.loc[restaurants["tz"].isna(), "state"].unique()
    if len(unmapped) > 0:
        raise ValueError(
            f"Unmapped states in STATE_TO_TZ: {sorted(unmapped)}. "
            f"Add IANA timezone entries before proceeding."
        )

    return restaurants

def filter_sparse_states(
    df: pd.DataFrame, min_restaurants: int = 100
) -> pd.DataFrame:
    """Remove restaurants in states with too few restaurants.

    Args:
        df: Restaurant DataFrame with 'state' column.
        min_restaurants: Minimum restaurants a state must have to keep.

    Returns:
        Filtered DataFrame.
    """
    state_counts = df["state"].value_counts()
    valid_states = state_counts[state_counts >= min_restaurants].index
    return df[df["state"].isin(valid_states)].copy()

# ── Review preprocessing ─────────────────────────────────────────────────────


def preprocess_reviews(
    df: pd.DataFrame,
    valid_business_ids: set,
    valid_user_ids: set,
    date_cutoff: str | None = "2020-01-01",
) -> pd.DataFrame:
    """Filter reviews to valid businesses/users and extract date features.

    Yelp review dates are YYYY-MM-DD only — no time-of-day.
    Time-of-day context comes from checkin data, not reviews.

    Args:
        df: Raw review DataFrame.
        valid_business_ids: Set of restaurant business_ids to keep.
        valid_user_ids: Set of user_ids to keep.
        date_cutoff: Exclude reviews on or after this date (e.g. "2020-03-01"
                     to remove pandemic-distorted data). None to disable.

    Returns DataFrame with all original columns plus:
        day_of_week, is_weekend, dow_sin, dow_cos
    """
    filtered = df[
        df["business_id"].isin(valid_business_ids)
        & df["user_id"].isin(valid_user_ids)
    ].copy()

    filtered["date"] = pd.to_datetime(filtered["date"])
    filtered = _apply_date_cutoff(filtered, "date", date_cutoff)

    filtered["day_of_week"] = filtered["date"].dt.dayofweek
    filtered["is_weekend"] = filtered["day_of_week"].isin([5, 6])

    # Cyclical encoding for day of week (period=7)
    angle = 2 * np.pi * filtered["day_of_week"] / 7.0
    filtered["dow_sin"] = np.sin(angle)
    filtered["dow_cos"] = np.cos(angle)

    return filtered

# ── Checkin-to-review matching ───────────────────────────────────────────────


def match_checkins_to_reviews(
    reviews: pd.DataFrame,
    checkins: pd.DataFrame,
    tolerance_days: int = 1,
) -> pd.DataFrame:
    """Approximate visit time by matching nearest prior checkin to each review.

    Yelp checkins are anonymous (no user_id), so matching is by business_id
    only. The nearest checkin at the same business within the tolerance
    window is used as a proxy for the reviewer's visit time.

    Both inputs must already be filtered to valid businesses.

    Args:
        reviews: Review DataFrame with 'review_id', 'business_id', 'date'.
        checkins: Checkin DataFrame with 'business_id', 'timestamp'.
        tolerance_days: Max days before review to look for a checkin match.

    Returns:
        Reviews DataFrame with added 'checkin_ts' column (NaT where no
        match found within tolerance).
    """
    r = reviews.sort_values("date")
    c = checkins[["business_id", "timestamp"]].copy().sort_values("timestamp")

    matched = pd.merge_asof(
        r,
        c,
        left_on="date",
        right_on="timestamp",
        by="business_id",
        direction="backward",
        tolerance=pd.Timedelta(days=tolerance_days),
    )

    return matched.rename(columns={"timestamp": "checkin_ts"})


def derive_checkin_time_features(
    reviews: pd.DataFrame,
    restaurants: pd.DataFrame,
) -> pd.DataFrame:
    """Derive cyclical time-of-visit features from matched checkin timestamps.

    Localizes UTC checkin_ts to the restaurant's timezone, then encodes
    hour-of-day and day-of-week as sin/cos pairs. Reviews without a
    matched checkin get zeros (model learns to handle partial signal).

    Args:
        reviews: Reviews DataFrame with 'checkin_ts' and 'business_id'.
        restaurants: Restaurants DataFrame with 'business_id' and 'tz'.

    Returns:
        Reviews DataFrame with added columns:
            checkin_hour_sin, checkin_hour_cos,
            checkin_dow_sin, checkin_dow_cos
    """
    reviews = reviews.copy()

    # Join timezone from restaurant metadata
    biz_tz = restaurants[["business_id", "tz"]].drop_duplicates("business_id")
    reviews = reviews.merge(biz_tz, on="business_id", how="left")
    reviews["tz"] = reviews["tz"].fillna(DEFAULT_TZ)

    # Localize matched checkin timestamps to local time
    has_checkin = reviews["checkin_ts"].notna()

    # Initialize with zeros (unmatched reviews stay zero)
    for col in ("checkin_hour_sin", "checkin_hour_cos",
                "checkin_dow_sin", "checkin_dow_cos"):
        reviews[col] = 0.0

    if has_checkin.any():
        matched = reviews.loc[has_checkin].copy()
        local_ts = localize_to_local(matched, "checkin_ts", "tz")

        hour = local_ts.dt.hour
        dow = local_ts.dt.dayofweek

        hour_angle = 2 * np.pi * hour / 24.0
        dow_angle = 2 * np.pi * dow / 7.0

        reviews.loc[has_checkin, "checkin_hour_sin"] = np.sin(hour_angle)
        reviews.loc[has_checkin, "checkin_hour_cos"] = np.cos(hour_angle)
        reviews.loc[has_checkin, "checkin_dow_sin"] = np.sin(dow_angle)
        reviews.loc[has_checkin, "checkin_dow_cos"] = np.cos(dow_angle)

    # Drop the joined tz column (it's on the restaurants table, not reviews)
    reviews = reviews.drop(columns=["tz"])

    return reviews


# ── User preprocessing ───────────────────────────────────────────────────────


def preprocess_users(df: pd.DataFrame, min_reviews: int = 5) -> pd.DataFrame:
    """Filter to users with sufficient review history.

    Note: Yelp dataset floor is 5 reviews per business, but user
    review counts start lower. Default min_reviews=5 aligns with
    the dataset's implicit floor.

    Returns DataFrame with all original columns, filtered.
    """
    return df[df["review_count"] >= min_reviews].copy()

# ── Interaction density filtering ────────────────────────────────────────────


def filter_sparse_interactions(
    reviews: pd.DataFrame,
    min_user_reviews: int = 10,
    min_business_reviews: int = 5,
) -> pd.DataFrame:
    """Iteratively remove users and businesses below interaction thresholds.

    A single pass can leave violations: removing sparse users may push
    a business below its threshold (and vice versa). This loops until
    convergence, typically 2-4 iterations.

    Uses a boolean mask rather than copying the DataFrame each iteration
    to keep memory flat on large review tables.

    Args:
        reviews: Review DataFrame with 'user_id' and 'business_id'.
        min_user_reviews: Minimum reviews a user must have in this dataset.
        min_business_reviews: Minimum reviews a business must have in this dataset.

    Returns:
        Filtered reviews DataFrame where all users and businesses meet
        their respective minimum interaction counts.
    """
    mask = np.ones(len(reviews), dtype=bool)

    while True:
        subset = reviews[mask]
        user_counts = subset["user_id"].value_counts()
        biz_counts = subset["business_id"].value_counts()

        valid_users = set(user_counts[user_counts >= min_user_reviews].index)
        valid_biz = set(biz_counts[biz_counts >= min_business_reviews].index)

        new_mask = (
            mask
            & reviews["user_id"].isin(valid_users)
            & reviews["business_id"].isin(valid_biz)
        )

        if new_mask.sum() == mask.sum():
            break
        mask = new_mask

    return reviews[mask].copy()


# ── Checkin preprocessing ────────────────────────────────────────────────────


def preprocess_checkins(
    df: pd.DataFrame,
    valid_business_ids: set,
    biz_tz: pd.DataFrame,
    date_cutoff: str | None = "2020-01-01",
) -> pd.DataFrame:
    """Build business-level temporal profiles from flattened checkin data.

    Expects extracted checkin parquet with columns:
        business_id (str), timestamp (datetime64)

    Timestamps are assumed UTC and converted to local time using the
    business timezone before extracting hour/day features.

    Produces pure distributions only — no derived statistics (entropy,
    peak hour, weekend ratio). The model's MLP has the capacity to learn
    whatever aggregations the data supports from the raw distributions.

    Args:
        df: Raw checkin DataFrame with one row per checkin event.
        valid_business_ids: Set of restaurant business_ids to keep.
        biz_tz: DataFrame with columns [business_id, tz] for timezone lookup.
        date_cutoff: Exclude checkins on or after this date (e.g. "2020-03-01"
                     to remove pandemic-distorted temporal patterns). None to disable.

    Returns DataFrame with columns:
        business_id, total_checkins,
        hour_dist_0 .. hour_dist_23  (fraction of checkins per hour)
        dow_dist_0 .. dow_dist_6     (fraction of checkins per day-of-week,
                                      0=Monday, 6=Sunday)
    """
    filtered = df[df["business_id"].isin(valid_business_ids)].copy()
    filtered = _apply_date_cutoff(filtered, "timestamp", date_cutoff)

    if len(filtered) == 0:
        return pd.DataFrame(columns=["business_id"])

    # Join timezone and convert UTC → local time
    filtered = filtered.merge(biz_tz[["business_id", "tz"]], on="business_id", how="left")
    filtered["tz"] = filtered["tz"].fillna(DEFAULT_TZ)
    filtered["timestamp"] = localize_to_local(filtered, "timestamp", "tz")

    filtered["hour"] = filtered["timestamp"].dt.hour
    filtered["day_of_week"] = filtered["timestamp"].dt.dayofweek

    # Per-business total
    grouped = filtered.groupby("business_id")
    profiles = pd.DataFrame()
    profiles["total_checkins"] = grouped.size()

    # Hour distribution (24 bins, normalized to fractions)
    hour_counts = (
        filtered.groupby(["business_id", "hour"])
        .size()
        .unstack(fill_value=0)
    )
    for h in range(24):
        if h not in hour_counts.columns:
            hour_counts[h] = 0
    hour_counts = hour_counts[sorted(hour_counts.columns)]
    hour_dist = hour_counts.div(hour_counts.sum(axis=1), axis=0)
    hour_dist.columns = [f"hour_dist_{h}" for h in range(24)]

    # Day-of-week distribution (7 bins, normalized to fractions)
    dow_counts = (
        filtered.groupby(["business_id", "day_of_week"])
        .size()
        .unstack(fill_value=0)
    )
    for d in range(7):
        if d not in dow_counts.columns:
            dow_counts[d] = 0
    dow_counts = dow_counts[sorted(dow_counts.columns)]
    dow_dist = dow_counts.div(dow_counts.sum(axis=1), axis=0)
    dow_dist.columns = [f"dow_dist_{d}" for d in range(7)]

    profiles = profiles.join(hour_dist).join(dow_dist)
    profiles = profiles.reset_index()

    return profiles


# ── Tip preprocessing ────────────────────────────────────────────────────────


def preprocess_tips(
    df: pd.DataFrame,
    valid_business_ids: set,
    biz_tz: pd.DataFrame,
    date_cutoff: str | None = "2020-01-01",
) -> pd.DataFrame:
    """Filter tips to valid businesses and extract features.

    Args:
        df: Raw tips DataFrame.
        valid_business_ids: Set of restaurant business_ids to keep.
        biz_tz: DataFrame with columns [business_id, tz] for timezone lookup.
        date_cutoff: Exclude tips on or after this date (e.g. "2020-03-01"
                     to remove pandemic-distorted data). None to disable.

    Returns DataFrame with all original columns filtered to valid businesses,
    plus day_of_week (derived from local time).
    """
    filtered = df[df["business_id"].isin(valid_business_ids)].copy()

    filtered["date"] = pd.to_datetime(filtered["date"])
    filtered = _apply_date_cutoff(filtered, "date", date_cutoff)

    # Tips only have date (no time), but localizing still corrects
    # edge cases where UTC midnight rolls to the previous day locally
    filtered = filtered.merge(biz_tz[["business_id", "tz"]], on="business_id", how="left")
    filtered["tz"] = filtered["tz"].fillna(DEFAULT_TZ)

    filtered["day_of_week"] = filtered["date"].dt.dayofweek

    return filtered