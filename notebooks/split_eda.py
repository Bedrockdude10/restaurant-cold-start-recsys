# %% [markdown]
# # Split Validation EDA
# Verify that train/val/test splits are well-formed:
# - No data leakage (cold start entities don't appear in train/val)
# - Temporal ordering is consistent
# - Distributions are reasonable
# - Cold start holdouts have enough signal to evaluate

# %%
import os
import pandas as pd
import numpy as np

os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # repo root

SPLITS_DIR = "data/splits"
PROCESSED_DIR = "data/processed"

# %%
train = pd.read_parquet(f"{SPLITS_DIR}/train.parquet")
val = pd.read_parquet(f"{SPLITS_DIR}/val.parquet")
test_warm = pd.read_parquet(f"{SPLITS_DIR}/test_warm.parquet")
test_cold_restaurant = pd.read_parquet(f"{SPLITS_DIR}/test_cold_restaurant.parquet")
test_cold_user = pd.read_parquet(f"{SPLITS_DIR}/test_cold_user.parquet")

cold_restaurant_ids = set(
    pd.read_csv(f"{SPLITS_DIR}/cold_start_restaurant_ids.csv")["business_id"]
)
cold_user_ids = set(
    pd.read_csv(f"{SPLITS_DIR}/cold_start_user_ids.csv")["user_id"]
)

restaurants = pd.read_parquet(f"{PROCESSED_DIR}/restaurants.parquet")

print("Loaded all splits and metadata.")

# %% [markdown]
# ## 1. Basic Counts

# %%
splits = {
    "train": train,
    "val": val,
    "test_warm": test_warm,
    "test_cold_restaurant": test_cold_restaurant,
    "test_cold_user": test_cold_user,
}

for name, df in splits.items():
    print(f"{name:25s}  {len(df):>8,} reviews  "
          f"{df['user_id'].nunique():>6,} users  "
          f"{df['business_id'].nunique():>6,} restaurants")

total = sum(len(df) for df in splits.values())
print(f"\n{'total':25s}  {total:>8,} reviews")

# %% [markdown]
# ## 2. Temporal Ordering
# Train dates should be strictly before val, which should be before test.

# %%
print("Date ranges:")
for name, df in splits.items():
    print(f"  {name:25s}  {df['date'].min()}  →  {df['date'].max()}")

# %%
# Check: no temporal leakage
assert train["date"].max() <= val["date"].min(), \
    "FAIL: train dates overlap with val!"
assert val["date"].max() <= test_warm["date"].min(), \
    "FAIL: val dates overlap with test!"
print("✅ Temporal ordering is clean — no overlap between train/val/test")

# %% [markdown]
# ## 3. Cold Start Leakage Checks
# The most critical validation: cold start entities must NOT appear in train or val.

# %%
# Cold start restaurants should not appear in train or val
train_restaurants = set(train["business_id"])
val_restaurants = set(val["business_id"])

leaked_restaurants_train = cold_restaurant_ids & train_restaurants
leaked_restaurants_val = cold_restaurant_ids & val_restaurants

assert len(leaked_restaurants_train) == 0, \
    f"FAIL: {len(leaked_restaurants_train)} cold start restaurants leaked into train!"
assert len(leaked_restaurants_val) == 0, \
    f"FAIL: {len(leaked_restaurants_val)} cold start restaurants leaked into val!"
print(f"✅ No cold start restaurants in train (checked {len(cold_restaurant_ids)} IDs)")
print(f"✅ No cold start restaurants in val")

# %%
# Cold start users should not appear in train or val
train_users = set(train["user_id"])
val_users = set(val["user_id"])

leaked_users_train = cold_user_ids & train_users
leaked_users_val = cold_user_ids & val_users

assert len(leaked_users_train) == 0, \
    f"FAIL: {len(leaked_users_train)} cold start users leaked into train!"
assert len(leaked_users_val) == 0, \
    f"FAIL: {len(leaked_users_val)} cold start users leaked into val!"
print(f"✅ No cold start users in train (checked {len(cold_user_ids)} IDs)")
print(f"✅ No cold start users in val")

# %%
# Cold restaurant test set should ONLY contain cold start restaurants
# reviewed by WARM users (not cold start users)
assert test_cold_restaurant["business_id"].isin(cold_restaurant_ids).all(), \
    "FAIL: test_cold_restaurant contains non-cold-start restaurants!"
cold_rest_users = set(test_cold_restaurant["user_id"])
leaked_cold_users_in_rest = cold_rest_users & cold_user_ids
assert len(leaked_cold_users_in_rest) == 0, \
    f"FAIL: {len(leaked_cold_users_in_rest)} cold start users in cold restaurant test set!"
print("✅ Cold restaurant test set: all restaurants are cold, all users are warm")

# %%
# Cold user test set should ONLY contain cold start users
# reviewing WARM restaurants (not cold start restaurants)
assert test_cold_user["user_id"].isin(cold_user_ids).all(), \
    "FAIL: test_cold_user contains non-cold-start users!"
cold_user_restaurants = set(test_cold_user["business_id"])
leaked_cold_rest_in_user = cold_user_restaurants & cold_restaurant_ids
assert len(leaked_cold_rest_in_user) == 0, \
    f"FAIL: {len(leaked_cold_rest_in_user)} cold start restaurants in cold user test set!"
print("✅ Cold user test set: all users are cold, all restaurants are warm")

# %% [markdown]
# ## 4. Cold Start Coverage
# How many held-out entities actually have test reviews we can evaluate against?

# %%
cold_restaurants_with_reviews = test_cold_restaurant["business_id"].nunique()
cold_users_with_reviews = test_cold_user["user_id"].nunique()

print(f"Cold start restaurants: {cold_restaurants_with_reviews} / "
      f"{len(cold_restaurant_ids)} have test reviews "
      f"({cold_restaurants_with_reviews / len(cold_restaurant_ids) * 100:.1f}%)")
print(f"Cold start users:      {cold_users_with_reviews} / "
      f"{len(cold_user_ids)} have test reviews "
      f"({cold_users_with_reviews / len(cold_user_ids) * 100:.1f}%)")

if cold_restaurants_with_reviews / len(cold_restaurant_ids) < 0.5:
    print("⚠️  Low restaurant cold start coverage — many held-out restaurants "
          "have no test-period reviews. Consider adjusting selection strategy.")
if cold_users_with_reviews / len(cold_user_ids) < 0.5:
    print("⚠️  Low user cold start coverage — many held-out users "
          "have no test-period reviews. Consider adjusting selection strategy.")

# %%
# Distribution of test reviews per cold start entity
cold_rest_reviews_per = test_cold_restaurant.groupby("business_id").size()
cold_user_reviews_per = test_cold_user.groupby("user_id").size()

print("Test reviews per cold start restaurant:")
print(cold_rest_reviews_per.describe())
print(f"\nTest reviews per cold start user:")
print(cold_user_reviews_per.describe())

# %% [markdown]
# ## 5. Star Rating Distributions
# Check that splits have similar rating distributions (no skew from the split).

# %%
print("Star rating distributions:")
for name, df in splits.items():
    dist = df["stars"].value_counts(normalize=True).sort_index()
    dist_str = "  ".join(f"{s}★:{v:.1%}" for s, v in dist.items())
    print(f"  {name:25s}  {dist_str}")

# %% [markdown]
# ## 6. User/Restaurant Overlap Between Train and Test
# For warm evaluation, most test users and restaurants should appear in train.

# %%
test_warm_users = set(test_warm["user_id"])
test_warm_restaurants = set(test_warm["business_id"])

user_overlap = len(test_warm_users & train_users) / len(test_warm_users) * 100
restaurant_overlap = len(test_warm_restaurants & train_restaurants) / len(test_warm_restaurants) * 100

print(f"Warm test users also in train:       {user_overlap:.1f}%")
print(f"Warm test restaurants also in train:  {restaurant_overlap:.1f}%")

if user_overlap < 80:
    print("⚠️  Low user overlap — many warm test users have no training history")
if restaurant_overlap < 80:
    print("⚠️  Low restaurant overlap — many warm test restaurants have no training history")

# %% [markdown]
# ## 7. Cold Start Restaurant Feature Coverage
# Verify held-out restaurants have the features the item tower needs.

# %%
cold_rest_features = restaurants[restaurants["business_id"].isin(cold_restaurant_ids)]
print(f"Cold start restaurants with feature data: "
      f"{len(cold_rest_features)} / {len(cold_restaurant_ids)}")

print(f"\nPrice tier distribution (cold start restaurants):")
print(cold_rest_features["price_tier"].value_counts().sort_index())

print(f"\nCategories null rate: "
      f"{cold_rest_features['categories'].isna().mean():.1%}")
print(f"Attributes null rate: "
      f"{cold_rest_features['attributes'].isna().mean():.1%}")
print(f"Hours null rate:      "
      f"{cold_rest_features['hours'].isna().mean():.1%}")

# Check if checkin profiles exist for cold start restaurants
try:
    checkins = pd.read_parquet(f"{PROCESSED_DIR}/checkin_profiles.parquet")
    cold_with_checkins = checkins[checkins["business_id"].isin(cold_restaurant_ids)]
    print(f"\nCheckin profiles available: "
          f"{len(cold_with_checkins)} / {len(cold_restaurant_ids)} "
          f"({len(cold_with_checkins) / len(cold_restaurant_ids) * 100:.1f}%)")
except FileNotFoundError:
    print("\nNo checkin_profiles.parquet found — skipping")

# %% [markdown]
# ## 8. Summary

# %%
all_passed = True
checks = [
    ("Temporal ordering", True),  # would have asserted above
    ("No cold restaurant leakage", len(leaked_restaurants_train) == 0 and len(leaked_restaurants_val) == 0),
    ("No cold user leakage", len(leaked_users_train) == 0 and len(leaked_users_val) == 0),
    ("Cold restaurant test integrity", len(leaked_cold_users_in_rest) == 0),
    ("Cold user test integrity", len(leaked_cold_rest_in_user) == 0),
    ("Restaurant cold start coverage > 50%", cold_restaurants_with_reviews / len(cold_restaurant_ids) > 0.5),
    ("User cold start coverage > 50%", cold_users_with_reviews / len(cold_user_ids) > 0.5),
    ("Warm user overlap > 80%", user_overlap > 80),
    ("Warm restaurant overlap > 80%", restaurant_overlap > 80),
]

print("Validation Summary")
print("=" * 55)
for name, passed in checks:
    status = "✅" if passed else "❌"
    print(f"  {status} {name}")
    if not passed:
        all_passed = False

print()
if all_passed:
    print("All checks passed — splits are ready for training.")
else:
    print("Some checks failed — review warnings above before proceeding.")