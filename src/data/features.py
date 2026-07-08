"""Feature extraction for user context and restaurant items.

Each tower receives both content features (static identity) and context
features (situational/temporal signals). The dot product between tower
outputs learns to match user situations to restaurant profiles.

═══ RESTAURANT TOWER (Pair B: Ben & Danny) ════════════════════════════

Restaurant content features (what a restaurant IS):
    Static metadata that describes the restaurant independent of time.
    These alone enable cold start — a new restaurant with only a Yelp
    listing still produces a meaningful embedding.

    - Category encoding: multi-hot over ~150 fine-grained Yelp categories,
      with PAD/UNK tokens, stop-category filtering, and min_df thresholding.
      Vocab built from training restaurants only to prevent leakage.
    - Price tier: ordinal 1-4 from RestaurantsPriceRange2 (default 2).
    - Structured attributes: binary flags (reservations, outdoor seating, etc.),
      categorical one-hot (WiFi, alcohol, noise, attire), and nested dicts
      (GoodForMeal, Ambience). Handles Yelp's inconsistent encoding
      (bool, str, u'value' literals, stringified dicts).

Restaurant context features (WHEN a restaurant is popular):
    Temporal profiles derived from checkin data. These pair with the user
    tower's day-of-week context so the model learns time-aware matching
    (e.g., Saturday morning user ↔ weekend-heavy brunch spot).

    - Hour distribution: 24-bin normalized histogram of checkin hours
      (converted from UTC to local time via business timezone).
    - Weekend ratio: fraction of checkins on Saturday/Sunday.
      0.5 = evenly split, 0.7+ = heavily weekend-skewed.
    - Hour entropy: Shannon entropy over the hour distribution.
      Low = niche timing (brunch-only), high = all-day restaurant.

    Restaurants without checkin data receive a zero vector; the restaurant
    tower's projection layer learns to produce a neutral embedding for
    these cases. Coverage is ~90%+ so this is rare.

═══ USER TOWER (Pair A: Rohith & Antonio) ═════════════════════════════

User content features (who the user IS):
    Static or slowly-changing user attributes that persist across visits.
    These enable user cold start — a new user with only onboarding
    preferences still produces a meaningful embedding.

    - Preference Profile: Provides information about what types of food a user likes
      based on restaurants they have interacted with (e.g User often visits italian 
      these categories will have higher preference values for italian in their profile).
    - Preference Vector:  Corresponds with preference profile, it
      stores the information in a ordered list of categories called 
      PREFERENCE_FEATURES so input format is consistent.
    - Geographic centroid: Mean taken from longitude/latitude of restaurants
      that each user has interacted with computed in training interaction.
      Provides a general location for each user, which can be utilized
      by the model to promote nearer restaurants. 

User context features (the user's current situation):
    Situational signals that change per visit. These pair with the
    restaurant tower's temporal profile so the model can match a user's
    current context to restaurants that are popular at similar times.

    - Day-of-week encoding: cyclical sin/cos (period=7) so the model
      learns that Sunday ↔ Monday are adjacent, not maximally distant.
      Preserves full granularity vs. collapsing into coarse buckets.
    - Is-weekend: binary flag (Saturday/Sunday). Redundant with sin/cos
      in theory, but gives the model an explicit signal for the
      weekday/weekend split without needing to learn it from sin/cos.
    - Distance: haversine km from user centroid to candidate restaurant,
      capped at 200km. Gets its own projection layer so the model can
      learn a non-linear distance decay (2km vs 5km matters more than
      50km vs 53km).
"""

import ast
from collections import Counter

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans


# ── Attribute schema ─────────────────────────────────────────────────────────
# Defines the full set of structured features extracted from Yelp's freeform
# attributes dict. Three types:
#   - Binary: true/false flags
#   - Categorical: one-hot over known levels + unknown
#   - Nested: dicts of sub-keys (e.g. GoodForMeal -> {breakfast, lunch, ...})

BINARY_KEYS = [
    "RestaurantsReservations",
    "OutdoorSeating",
    "HasTV",
    "GoodForKids",
    "BusinessAcceptsCreditCards",
]

CATEGORICAL_SCHEMA = {
    "WiFi": ("wifi", ["free", "paid", "no"]),
    "Alcohol": ("alcohol", ["none", "beer_and_wine", "full_bar"]),
    "NoiseLevel": ("noise", ["quiet", "average", "loud", "very_loud"]),
    "RestaurantsAttire": ("attire", ["casual", "dressy", "formal"]),
}

NESTED_SCHEMA = {
    "GoodForMeal": ("meal", ["breakfast", "brunch", "lunch", "dinner", "latenight"]),
    "Ambience": ("ambience", ["casual", "romantic", "touristy", "hipster", "upscale"]),
}


def extract_attribute_features(attributes) -> dict[str, float]:
    """Extract structured features from a Yelp business attributes dict.

    Handles Yelp's inconsistent encoding: values may be bool, str,
    u'value' Python-literal strings, or nested dicts encoded as strings.

    Args:
        attributes: Raw attributes dict from the Yelp dataset (or None).

    Returns:
        Flat dict of feature_name -> float (0.0 or 1.0).
        Feature names follow the pattern:
            Binary:      attr_{key}
            Categorical: {prefix}_{level} + {prefix}_unknown
            Nested:      {prefix}_{sub_key}
    """
    attrs = attributes if isinstance(attributes, dict) else {}
    feats = {}

    # Binary flags
    for key in BINARY_KEYS:
        val = attrs.get(key)
        if isinstance(val, str):
            val = val.strip().strip("'\"").lower() in ("true", "1", "yes")
        feats[f"attr_{key}"] = float(bool(val))

    # Categorical (one-hot with unknown bucket)
    for attr_key, (prefix, levels) in CATEGORICAL_SCHEMA.items():
        value = attrs.get(attr_key)
        if isinstance(value, str):
            value = value.strip().strip("'\"").lower()
            # Strip Yelp's u'value' encoding
            if value.startswith("u'") or value.startswith('u"'):
                value = value[2:].rstrip("'\"")
            if value in ("none", "null", ""):
                value = None
        for level in levels:
            feats[f"{prefix}_{level}"] = float(value == level)
        feats[f"{prefix}_unknown"] = float(value is None or value not in levels)

    # Nested dicts (e.g. GoodForMeal, Ambience)
    for attr_key, (prefix, sub_keys) in NESTED_SCHEMA.items():
        nested = attrs.get(attr_key, {}) or {}
        if isinstance(nested, str):
            try:
                nested = ast.literal_eval(nested)
            except Exception:
                nested = {}
        if not isinstance(nested, dict):
            nested = {}
        for sub_key in sub_keys:
            feats[f"{prefix}_{sub_key}"] = float(bool(nested.get(sub_key)))

    return feats


# Canonical ordered list of attribute feature names (for consistent vectorization)
ATTR_NAMES: list[str] = list(extract_attribute_features(None).keys())
ATTR_DIM: int = len(ATTR_NAMES)


# ── Category vocabulary ──────────────────────────────────────────────────────

STOP_CATEGORIES = {
    "Restaurants", "Food", "Nightlife", "Bars",
    "Event Planning & Services", "Shopping",
}


def build_category_vocab(
    restaurants: pd.DataFrame,
    train_business_ids: set[str],
    min_df: int = 10,
) -> dict[str, int]:
    """Build category-to-index vocab from training restaurants only.

    Filters out generic stop categories and low-frequency categories.

    Args:
        restaurants: DataFrame with 'business_id' and 'categories' (list[str]).
        train_business_ids: Set of business_ids in the training split.
        min_df: Minimum document frequency to include a category.

    Returns:
        Dict mapping category string -> integer index.
        Index 0 = PAD, 1 = UNK, 2+ = categories sorted by frequency.
    """
    train_biz = restaurants[restaurants["business_id"].isin(train_business_ids)]
    counter: Counter[str] = Counter()
    for cats in train_biz["categories"]:
        if isinstance(cats, (list, np.ndarray)):
            counter.update(set(cats) - STOP_CATEGORIES)

    vocab = {"<PAD>": 0, "<UNK>": 1}
    for cat, count in counter.most_common():
        if count >= min_df:
            vocab[cat] = len(vocab)
    return vocab


def encode_categories(
    cats, vocab: dict[str, int], max_len: int = 15
) -> list[int]:
    """Encode a list of category strings to integer IDs.

    Filters stop categories, deduplicates, and maps unseen categories
    to UNK. Returns unpadded list (caller handles padding).

    Args:
        cats: List/array of category strings, or non-iterable.
        vocab: Category vocabulary from build_category_vocab.
        max_len: Maximum number of category IDs to return.

    Returns:
        List of integer category IDs (length <= max_len).
    """
    if not isinstance(cats, (list, np.ndarray)):
        return [1]  # UNK
    ids = []
    seen: set[int] = set()
    for c in cats:
        if c in STOP_CATEGORIES:
            continue
        idx = vocab.get(c, 1)  # UNK for unseen
        if idx not in seen:
            ids.append(idx)
            seen.add(idx)
    return ids[:max_len] if ids else [1]


# ── User features ────────────────────────────────────────────────────────────


def extract_cuisine_preferences(
    user_id: str,
    reviews: pd.DataFrame,
    businesses: pd.DataFrame,
    top_k: int = 3,
) -> list[str]:
    """Extract a user's top-k cuisine categories from their review history.

    Used to simulate onboarding preferences for cold start evaluation:
    mask a user's reviews, keep only their top cuisine categories.
    """
    user_reviews = reviews[reviews["user_id"] == user_id]
    reviewed_businesses = businesses[
        businesses["business_id"].isin(user_reviews["business_id"])
    ]

    # Flatten all categories, count occurrences
    all_cats = []
    for cats in reviewed_businesses["categories_list"]:
        if isinstance(cats, list):
            all_cats.extend(cats)

    if not all_cats:
        return []

    cat_counts = pd.Series(all_cats).value_counts()
    return cat_counts.head(top_k).index.tolist()


def encode_time_features(hour: int, day_of_week: int) -> dict:
    """Encode time context into feature dict for the user tower.

    Uses cyclical sin/cos encoding so the model learns that hour 23 and
    hour 0 are adjacent (not maximally distant) and retains full hourly
    granularity instead of collapsing into coarse buckets.

    Args:
        hour: Hour of day (0-23).
        day_of_week: Day of week (0=Monday, 6=Sunday).

    Returns:
        dict with keys: hour_sin, hour_cos, day_sin, day_cos, is_weekend
    """
    hour_rad = 2 * np.pi * hour / 24
    day_rad = 2 * np.pi * day_of_week / 7

    return {
        "hour_sin": float(np.sin(hour_rad)),
        "hour_cos": float(np.cos(hour_rad)),
        "day_sin": float(np.sin(day_rad)),
        "day_cos": float(np.cos(day_rad)),
        "is_weekend": day_of_week in (5, 6),
    }


def compute_user_centroids(
    reviews: pd.DataFrame,
    businesses: pd.DataFrame,
) -> dict[str, tuple[float, float]]:
    """Compute geographic centroid for each user based on their review locations.

    Vectorized via merge + groupby (not per-user loops).

    Args:
        reviews: Reviews DataFrame with 'user_id' and 'business_id'.
        businesses: Businesses DataFrame with 'business_id', 'latitude', 'longitude'.

    Returns:
        dict mapping user_id -> (latitude, longitude)
    """
    biz_locs = businesses.set_index("business_id")[["latitude", "longitude"]]
    merged = reviews[["user_id", "business_id"]].merge(
        biz_locs, left_on="business_id", right_index=True, how="left",
    )
    grouped = merged.groupby("user_id")[["latitude", "longitude"]].mean().to_dict("index")
    return {str(uid): (float(v["latitude"]), float(v["longitude"])) for uid, v in grouped.items()}


def cluster_geographic_regions(
    businesses: pd.DataFrame,
    n_clusters: int = 50,
    random_state: int = 42,
) -> tuple[np.ndarray, KMeans]:
    """Assign businesses to geographic regions via K-Means on lat/lng.

    Returns:
        region_labels: array of cluster IDs for each business
        kmeans: fitted KMeans model (for assigning new restaurants at inference)
    """
    coords = businesses[["latitude", "longitude"]].dropna().values
    kmeans = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=10)
    labels = kmeans.fit_predict(coords)
    return labels, kmeans

# User preference features --- New Anto
# NOTE: These must match the casing in your Yelp categories as stored
# in the pipeline (title-case from extract_yelp.py). The lookup in
# compute_user_preferences lowercases both sides for matching.

CUISINES = {
    "Thai", "Japanese", "Chinese", "Mexican", "Italian", "Indian",
    "Korean", "Vietnamese", "French", "Greek", "Mediterranean",
    "American (Traditional)", "American (New)", "Southern", "Cajun/Creole",
    "Caribbean", "Latin American",
    "Pakistani", "Cuban",
    "Asian Fusion", "Tex-Mex", "Soul Food",
}

FOOD_TYPES = {
    "Pizza", "Ramen", "Sandwiches", "Seafood", "Steakhouses",
    "Sushi Bars", "Noodles", "Barbeque", "Breakfast & Brunch",
    "Coffee & Tea", "Bakeries", "Desserts", "Ice Cream & Frozen Yogurt",
    "Juice Bars & Smoothies", "Fast Food", "Diners", "Buffets",
    "Food Trucks", "Comfort Food", "Chicken Wings", "Tacos",
    "Hot Dogs", "Salad", "Soup",
}

# Lowercased versions for matching against normalized category strings
_CUISINES_LOWER = {c.lower() for c in CUISINES}
_FOOD_TYPES_LOWER = {c.lower() for c in FOOD_TYPES}

# Canonical feature names are lowercase+sorted for stable column ordering
PREFERENCE_FEATURES = tuple(sorted(_CUISINES_LOWER | _FOOD_TYPES_LOWER))

# Checkin-matched time-of-visit feature keys (from preprocessing)
CHECKIN_KEYS = frozenset({
    "checkin_hour_sin", "checkin_hour_cos",
    "checkin_dow_sin", "checkin_dow_cos",
})

def _build_preferences_table(user_visits: pd.DataFrame, categories_lower: set[str]) -> pd.DataFrame:
    """
    user_visits: columns [user_id, categories] where categories is one lowercase string per row.
    categories_lower: set of lowercase category strings to match against.
    Returns: DataFrame indexed by user_id with one col per category, normalized frequencies.
    """
    filtered = user_visits[user_visits["categories"].isin(categories_lower)]
    counts = filtered.pivot_table(index="user_id", columns="categories", aggfunc="size", fill_value=0)
    counts = counts.reindex(columns=sorted(categories_lower), fill_value=0)
    return counts.div(counts.sum(axis=1), axis=0).fillna(0.0)


def compute_user_preferences(interactions: pd.DataFrame, restaurants: pd.DataFrame) -> pd.DataFrame:
    """
    Compute per-user preference features from interaction history.
    interactions: [user_id, business_id]
    restaurants:  [business_id, categories] where categories is list[str]

    Categories are lowercased for matching against CUISINES/FOOD_TYPES,
    and output columns use the canonical lowercase PREFERENCE_FEATURES ordering.
    """
    biz_cats = restaurants[["business_id", "categories"]].copy()
    biz_cats = biz_cats.explode("categories").dropna(subset=["categories"])
    biz_cats["categories"] = biz_cats["categories"].astype(str).str.strip().str.lower()

    visits = interactions[["user_id", "business_id"]].copy()
    visits["user_id"] = visits["user_id"].astype(str)

    # one row per (user visit, category)
    user_visits = visits.merge(biz_cats, on="business_id", how="left").dropna(subset=["categories"])

    cuisine = _build_preferences_table(user_visits, _CUISINES_LOWER)
    food = _build_preferences_table(user_visits, _FOOD_TYPES_LOWER)
    prefs = pd.concat([cuisine, food], axis=1).fillna(0.0)

    all_users = visits["user_id"].unique()
    prefs.index = prefs.index.astype(str)
    prefs = prefs.reindex(all_users, fill_value=0.0)
    prefs = prefs.reindex(columns=list(PREFERENCE_FEATURES), fill_value=0.0)
    return prefs


# ── Onboarding simulation ───────────────────────────────────────────────────
# Simulates the cold-start onboarding flow where a new user selects 1-5
# favorite cuisines. For training users, selections are sampled proportional
# to their actual review-frequency distribution over cuisine/food-type
# categories, introducing realistic noise (users don't perfectly report
# their habits). Per-epoch resampling acts as free data augmentation.


def _sample_onboarding_array(
    probs: np.ndarray,
    max_k: int = 5,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Vectorized onboarding simulation via Gumbel-max sampling.

    For each user, samples k ~ Uniform(1, max_k) categories without
    replacement, proportional to their cuisine frequency distribution.
    Uses the Gumbel-max trick: adding Gumbel noise to log-probabilities
    and taking the top-k gives exact samples from the categorical
    distribution without replacement — fully vectorized, no per-user loops.

    Users with no preference history (all-zero rows) get a zero vector,
    matching the true cold-start scenario where we have no signal at all.

    Args:
        probs: (n_users, n_features) frequency distributions (row-normalized
            or raw counts — will be normalized internally).
        max_k: Maximum categories per user (k drawn from Uniform(1, max_k)).
        rng: NumPy random generator for reproducibility.

    Returns:
        (n_users, n_features) float32 array with 1.0 at selected categories.
    """
    if rng is None:
        rng = np.random.default_rng()

    probs = probs.copy().astype(np.float64)
    n_users, n_features = probs.shape

    # Normalize rows to valid probability distributions
    row_sums = probs.sum(axis=1, keepdims=True)
    valid = row_sums.squeeze() > 0
    probs[valid] /= row_sums[valid]

    # k per user: Uniform(1, max_k), capped at available non-zero categories
    ks = rng.integers(1, max_k + 1, size=n_users)
    available = (probs > 0).sum(axis=1).astype(int)
    ks = np.minimum(ks, available)

    # Gumbel-max trick: log(prob) + Gumbel noise, then argsort descending.
    # Top-k of the perturbed scores = exact sample without replacement.
    log_probs = np.full_like(probs, -np.inf)
    log_probs[valid] = np.log(probs[valid] + 1e-10)
    gumbel = rng.gumbel(size=(n_users, n_features))
    perturbed = log_probs + gumbel

    sorted_idx = np.argsort(-perturbed, axis=1)  # (n_users, n_features)

    # Selection mask: position j is selected if j < k[i]
    positions = np.arange(n_features)[np.newaxis, :]  # (1, n_features)
    select_mask = positions < ks[:, np.newaxis]        # (n_users, n_features)

    # Scatter selections back to original feature positions
    result = np.zeros((n_users, n_features), dtype=np.float32)
    rows = np.repeat(np.arange(n_users), n_features)
    cols = sorted_idx.ravel()
    mask_flat = select_mask.ravel()
    result[rows[mask_flat], cols[mask_flat]] = 1.0

    return result


def sample_onboarding_selections(
    user_preferences: pd.DataFrame,
    max_k: int = 5,
    rng: np.random.Generator | None = None,
) -> pd.DataFrame:
    """Sample binary onboarding vectors from user preference distributions.

    DataFrame wrapper around _sample_onboarding_array for use in
    pipeline.py and evaluate.py where index/column alignment matters.

    Args:
        user_preferences: DataFrame indexed by user_id, columns=PREFERENCE_FEATURES,
            values are normalized visit frequencies (from compute_user_preferences).
        max_k: Maximum cuisines a user can select during onboarding.
        rng: NumPy random generator.

    Returns:
        DataFrame with same index/columns as user_preferences, binary {0, 1} values.
    """
    result = _sample_onboarding_array(user_preferences.values, max_k=max_k, rng=rng)
    return pd.DataFrame(result, index=user_preferences.index, columns=user_preferences.columns)