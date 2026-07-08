# Restaurant Cold-Start RecSys

Research code and ablation study for a **context-aware two-tower recommender** that
tackles two cold-start problems in restaurant recommendation *simultaneously*:

1. **New users** — only onboarding cuisine selections plus situational context
   (day-of-week, distance), no rating history.
2. **New restaurants** — only metadata (categories, price, attributes, and a
   check-in–derived temporal profile), no reviews.

Both towers encode **features, not IDs**, so a brand-new user or restaurant still
produces a meaningful embedding. The model is trained and evaluated on the
[Yelp Open Dataset](https://www.yelp.com/dataset) (~1.3M reviews, ~35K restaurants,
~59K users after filtering).

> This repository accompanies a paper draft currently under submission. The dataset
> itself is **not redistributed** here — see [`data/README.md`](data/README.md) to
> obtain it. Everything committed is source code and *aggregate* evaluation results.

---

## Key results

Ranking is over one held-out positive plus **same-city** sampled negatives
(~100 candidates; random floor ≈ 5%). Metrics are **Hit@5** and **NDCG@10**.
Values are the best two-tower variant per split, compared against Random, Popularity, and
**DropoutNet** (Volkovs et al., 2017) — a purpose-built cold-start baseline, run on the
*same* frozen test cases (see [Baselines](#baselines)).

| Split | Cases | Random H@5 / N@10 | Popularity H@5 / N@10 | DropoutNet H@5 / N@10 | Two-Tower H@5 / N@10 |
|-------|------:|:---:|:---:|:---:|:---:|
| **Warm** | 80,057 | 5.4 / 5.0 | 27.2 / 23.2 | 18.0 / 15.7 | **28.7 / 24.3** |
| **Cold-Restaurant** | 1,369 | 4.1 / 3.8 | 0.0 / 0.0 | **12.5 / 9.8** | 10.0 / 9.3 |
| **Cold-User** | 1,732 | 6.2 / 5.4 | 27.1 / 22.8 | 11.8 / 10.6 | **31.8 / 29.0** |

*(all numbers in %; best per split in bold)*

**How to read this honestly:**

- **Cold restaurants — content wins, and so does DropoutNet.** Popularity is structurally
  **0** (held-out restaurants have no training reviews). DropoutNet, a method built for cold
  start, is the *strongest* here (12.5 vs. 10.0 Hit@5); our two-tower is competitive but does
  **not** beat it. Content is the decisive signal for cold items.
- **Cold users — context wins, decisively.** The two-tower reaches 29.0 NDCG@10 vs.
  Popularity's 22.8 and DropoutNet's 10.6. DropoutNet, lacking situational features, falls
  below even Popularity — the gap the context sub-towers close, and this model's distinctive
  contribution.
- **Warm — Popularity is hard to beat.** The two-tower ties it (24.3 vs. 23.2); DropoutNet
  (lightweight SVD backbone) trails. The personalized win is concentrated in cold start.

One-line thesis: **content for cold items, context for cold users.**

---

## Model

### Two-tower (dual encoder), four sub-towers

Each side is split into a **content** sub-tower ("what/who it is") and a **context**
sub-tower ("when/where"). The two 64-d sub-tower outputs are concatenated and combined
by a fusion MLP into a final 64-d embedding. Scoring is a dot product (cosine optional),
trained with **BPR loss** against **geographically-sampled negatives**.

**User tower**
- *Content* — preference vector (normalized visit-frequency over 47 curated cuisines) +
  onboarding vector (multi-hot of 1–5 simulated cuisine picks, sampled via the Gumbel-max
  trick and resampled every epoch as augmentation).
- *Context* — day-of-week (cyclical sin/cos + weekend flag) + Haversine distance from the
  user's geographic centroid (capped at 200 km).

**Restaurant tower**
- *Content* — categories (mean-pooled learned embeddings over the Yelp category vocab) +
  price tier (ordinal 1–4) + structured attributes (reservations, WiFi, alcohol, ambience,
  …) projected to 16-d.
- *Context* — temporal profile: 24-bin hour distribution + weekend ratio + hour entropy,
  derived from check-ins (UTC→local via per-state IANA timezones), projected to 16-d.

### Fusion variants
- **`FusionMLP`** — concatenate content+context, then MLP.
- **`GatedFusionMLP`** — content-conditioned gating of the context signal (inspired by
  content-gated feature-interaction work for cold-start).

### Training signal for cold restaurants
During training the raw restaurant `temporal_vec` is zeroed for 20% of samples
(**input-level temporal dropout**), directly simulating the cold-restaurant scenario so
the model does not become dependent on a signal that is often missing at inference.

---

## Data pipeline

Four stages (see [`scripts/`](scripts/)):

1. **Extract** (`extract_yelp.py`) — raw Yelp JSON → typed parquet, no filtering.
2. **Preprocess** (`preprocess_yelp.py`) — filter to restaurants, extract price/attributes,
   build per-restaurant check-in temporal profiles (UTC→local), density-filter users and
   restaurants, apply a pre-COVID date cutoff (default 2020-01-01).
3. **Split** (`create_splits.py`) — stratified **cold-start holdout** of users and
   restaurants (all their reviews removed from train/val), then a temporal 80/10/10 split of
   the remaining warm reviews. Both-cold reviews are dropped.
4. **Feature prep** (`src/data/pipeline.py`) — `prepare_features()` is the single entry
   point shared by training and evaluation (category vocab, business feature aggregation,
   city-level geo indexes for negative sampling, user centroids/preferences, initial
   onboarding sample).

Dataset after filtering: **1,293,531** train reviews · **58,801** users ·
**35,565** restaurants (14 US/CA states; sparsity 0.99938). Test cases:
80,057 warm · 1,369 cold-restaurant · 1,732 cold-user.

---

## Evaluation

For each test interaction (user rated restaurant *X* with stars ≥ threshold):
sample *N* negatives **from the same city**, score the 1 positive + *N* negatives with
each model/baseline, rank, and compute Hit@5 / NDCG@10. Test cases are **frozen** across
models and epochs for fair comparison; training negatives are resampled every epoch.

- **Cold-user leave-one-out onboarding** — a cold user's onboarding vector is built from
  their *test* reviews *excluding* the one containing the ground-truth positive, so the
  label never leaks into the input (`build_loo_onboarding()`).
### Baselines

- **Random** (uniform) and **Popularity** (rank by training review count).
- **DropoutNet** (Volkovs et al., NeurIPS 2017) — a learned cold-start baseline, run on the
  *same* frozen test cases (`scripts/dropoutnet_baseline.py` →
  `results/dropoutnet_results.json`). It uses `k=64` collaborative latent factors from a
  truncated SVD of the binary train interaction matrix (a lightweight WMF substitute) plus
  the same content features the two-tower uses (user preference/onboarding; item categories +
  price), and is trained to reconstruct collaborative affinity with per-sample latent
  input-dropout — so content carries the signal when a latent is missing. At inference the
  user latent is zeroed for cold users and the item latent for all candidates in the
  cold-restaurant split (content-only ranking), mirroring the two-tower's protocol. Recomputed
  Popularity matches the reference numbers to four decimals, confirming identical test cases.

---

## Ablation study

We vary which **feature groups** are active on each tower, then test targeted feature and
architecture changes. Twelve experiments in total.

### 1. Feature-group grid (which half of each tower is active)

|                     | Restaurant: All | Restaurant: Content-only | Restaurant: Context-only |
|---------------------|-----------------|--------------------------|--------------------------|
| **User: All**       | `full_model`    | `no_temporal`            | *(not run)*              |
| **User: Content**   | `no_user_context` | `content_x_content`    | *(not run)*              |
| **User: Context**   | `no_user_content` | *(not run)*            | `context_x_context`      |

> The three "restaurant context-only" cells other than `context_x_context` were not run;
> they are noted as future work. (An earlier draft of this table labeled cells
> `no_restaurant_context` / `no_restaurant_content`; the experiment actually run is
> `no_temporal`, which removes the restaurant context sub-tower.)

### 2. User-content sub-ablations
`no_onboarding` (preferences only) and `no_preferences` (onboarding only).

### 3. Architecture / feature variants
`with_user_checkin`, `checkin_replaces_dow`, `gated_fusion`, `gated_fusion_with_checkin`.

### Full results

All values %. Best per column in **bold**. Baselines (constant): Warm 5.4/5.0 (Random),
27.2/23.2 (Pop); Cold-Rest 4.1/3.8, 0.0/0.0; Cold-User 6.2/5.4, 27.1/22.8.

| Experiment | Warm H@5 | Warm N@10 | ColdRest H@5 | ColdRest N@10 | ColdUser H@5 | ColdUser N@10 |
|------------|---------:|----------:|-------------:|--------------:|-------------:|--------------:|
| full_model | 26.5 | 22.6 | 9.0 | **9.3** | 29.1 | 24.8 |
| no_user_context | 26.5 | 22.6 | 7.0 | 6.8 | 24.9 | 21.7 |
| content_x_content | 25.0 | 21.4 | 5.0 | 4.7 | 24.2 | 20.7 |
| no_user_content | 27.2 | 23.1 | 3.9 | 3.6 | 31.5 | **29.0** |
| context_x_context | 27.5 | 23.5 | 2.5 | 2.5 | **31.8** | 29.0 |
| checkin_replaces_dow | 28.0 | 23.8 | **10.0** | 8.5 | 31.8 | 27.0 |
| gated_fusion | 27.7 | 23.5 | 7.7 | 7.2 | 29.1 | 25.4 |
| gated_fusion_with_checkin | **28.7** | 24.3 | 8.8 | 7.5 | 31.5 | 26.6 |
| no_onboarding | 27.7 | 23.4 | 7.0 | 7.1 | 29.2 | 25.7 |
| no_preferences | 27.9 | 23.6 | 3.9 | 3.7 | **31.8** | 28.9 |
| no_temporal | 26.1 | 22.4 | 6.7 | 6.1 | 28.2 | 24.3 |
| with_user_checkin | 28.6 | **24.3** | 9.1 | 8.0 | 31.4 | 26.7 |

Reproduced from [`results/ablation/summary.csv`](results/ablation/summary.csv). Per-experiment
metrics are under [`results/ablation/<experiment>/eval_results.json`](results/ablation/); an
interactive summary is in [`results/dashboard.html`](results/dashboard.html).

### Findings
- **Content is essential for cold restaurants.** Dropping user content collapses
  cold-restaurant NDCG@10 from 9.3 → 3.6; context alone is nearly useless (2.5).
- **Context drives cold users.** `context_x_context` (no content on either side) *beats*
  the full model on cold users (29.0 vs 24.8 NDCG@10) — sparse onboarding is noisier than
  situational signal.
- **Check-in–matched visit times help broadly.** `with_user_checkin` lifts warm NDCG@10
  22.6 → 24.3 and cold-user 24.8 → 26.7; `checkin_replaces_dow` gives the best
  cold-restaurant Hit@5 (10.0).
- **Gated fusion helps warm/cold-user but hurts cold-restaurant** — it learns
  training-restaurant-specific interactions that don't transfer to unseen restaurants.
- **Onboarding vs. preferences are asymmetric.** Dropping preferences hurts cold
  restaurants (9.3 → 3.7) but *improves* cold users (24.8 → 28.9).

---

## Setup

```bash
git clone https://github.com/Bedrockdude10/restaurant-cold-start-recsys.git
cd restaurant-cold-start-recsys

python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"

# Obtain the Yelp Open Dataset (not redistributed here) — see data/README.md
```

Then run the pipeline (see [`data/README.md`](data/README.md) for the full command list):

```bash
python scripts/extract_yelp.py       --raw-dir data/raw --out-dir data/extracted
python scripts/preprocess_yelp.py    --config configs/default.yaml
python scripts/create_splits.py      --config configs/default.yaml
python scripts/train.py              --data-dir data --epochs 50
python scripts/run_ablation.py       --data-dir data --epochs 50
python scripts/summarize_ablation.py --results-dir results/ablation --csv results/ablation/summary.csv
```

---

## Repository layout

```
restaurant-cold-start-recsys/
├── configs/default.yaml        # all hyperparameters
├── data/README.md              # how to obtain the Yelp dataset (data is gitignored)
├── scripts/                    # extract → preprocess → split → train → evaluate → ablate
├── src/
│   ├── data/                   # features, dataset, shared pipeline, preprocessing
│   ├── models/two_tower.py     # towers, FusionMLP / GatedFusionMLP, TwoTowerModel
│   ├── evaluation/             # metrics, geo sampling, baselines, cold-start protocols
│   └── utils/                  # config, Haversine geo
├── notebooks/                  # EDA + preprocessing + evaluation (outputs cleared)
├── results/ablation/           # summary.csv + per-experiment eval_results.json (aggregate only)
└── tests/                      # pytest: features, metrics, two-tower
```

---

## Team

| Member | Pair | Focus |
|--------|------|-------|
| Rohith | A — User Tower | Cuisine preference embeddings, onboarding simulation, user feature integration |
| Antonio | A — User Tower | Yelp user preprocessing, cuisine/food-type extraction, preference computation, cold-start eval |
| Ben | B — Restaurant Tower | Category embeddings, price-tier encoding, attribute extraction, tower training |
| Danny | B — Restaurant Tower | Yelp business preprocessing, check-in temporal profiles, geographic sampling, pipeline architecture |

Originally developed for CS 7180 (Applied Deep Learning), Spring 2026.

## Citation

If you use this code or its results, please cite the accompanying paper (details to
follow on acceptance):

```bibtex
@misc{restaurant_coldstart_2026,
  title  = {Content for Cold Items, Context for Cold Users: A Two-Tower Study of the Dual Cold-Start Problem in Restaurant Recommendation},
  author = {Rollo, Danny and Lin, Benjamin and Tagliatti, Antonio and Rohith},
  year   = {2026},
  note   = {Under submission}
}
```

## License

Code is released under the [MIT License](LICENSE). The Yelp Open Dataset is subject to
[Yelp's Dataset License](https://www.yelp.com/dataset) and is not included in this repository.
