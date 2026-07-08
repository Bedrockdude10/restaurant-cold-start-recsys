# Data

This project uses the **[Yelp Open Dataset](https://www.yelp.com/dataset)**. Per Yelp's
Dataset License, the data is **not redistributed** in this repository — you must download
it yourself. Everything in this `data/` directory (other than this README) is gitignored.

## 1. Download

1. Go to <https://www.yelp.com/dataset> and accept the Dataset License.
2. Download the JSON archive and extract it so the raw files live under `data/raw/`:

```
data/raw/
├── yelp_academic_dataset_business.json
├── yelp_academic_dataset_review.json
├── yelp_academic_dataset_user.json
├── yelp_academic_dataset_checkin.json
└── yelp_academic_dataset_tip.json
```

## 2. Build the processed data and splits

From the repository root, with the environment installed (`pip install -e ".[dev]"`):

```bash
# Stage 1 — raw JSON → typed parquet (no filtering)
python scripts/extract_yelp.py    --raw-dir data/raw --out-dir data/extracted

# Stage 2 — filter to restaurants, encode features, build check-in temporal profiles
python scripts/preprocess_yelp.py --config configs/default.yaml

# Stage 3 — cold-start holdout + temporal 80/10/10 split
python scripts/create_splits.py   --config configs/default.yaml
```

This produces `data/processed/` and `data/splits/`. See
[`configs/default.yaml`](../configs/default.yaml) for all knobs (date cutoff, minimum
interaction counts, split ratios, etc.).

## Expected scale (reference)

After the default filtering (14 US/CA states, pre-2020 cutoff):

| Split | Reviews | Users | Restaurants |
|-------|--------:|------:|------------:|
| Train | 1,293,531 | 58,801 | 35,565 |
| Val | 161,691 | — | — |
| Test — warm | 97,844 → 80,057 cases | 22,627 | 20,190 |
| Test — cold-restaurant | 1,682 → 1,369 cases | — | 289 (500 held out) |
| Test — cold-user | 2,087 → 1,732 cases | 426 (1,000 held out) | — |

Exact counts depend on the Yelp dataset release you download.
