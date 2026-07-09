"""Single source of truth for the ablation study experiments.

Every consumer — the runner (``scripts/run_ablation.py``), the summarizer
(``scripts/summarize_ablation.py``), and the analysis notebook
(``notebooks/evaluate_ablation.ipynb``) — imports the experiment registry from
here. Nothing else defines the set of experiments, their feature groups, labels,
display order, or key comparisons, so the runner and the reporting layers can
never drift out of sync.

To add / change an experiment, edit ``EXPERIMENTS`` (and, if it belongs in the
reporting views, ``DISPLAY_ORDER`` / ``KEY_COMPARISONS``) here and nowhere else.

Feature groups (must match the choices accepted by ``scripts/train.py``):
    Restaurant content: attributes, categories, price
    Restaurant context: temporal
    User context:       day_of_week, distance
    User content:       onboarding, preferences
    User context (opt): user_checkin
"""

from dataclasses import dataclass, field


@dataclass(frozen=True)
class Experiment:
    """One ablation configuration.

    Attributes:
        name:              Stable identifier — also the results subdirectory name.
        label:             Long, descriptive label for tables / paper figures.
        restaurant_groups: Enabled restaurant-tower feature groups.
        user_groups:       Enabled user-tower feature groups.
        extra_args:        Extra CLI args passed through to ``train.py`` (e.g. fusion).
        short_label:       Compact label for plot axes / legends.
        restaurant_desc:   Human summary of the restaurant tower (matrix view).
        user_desc:         Human summary of the user tower (matrix view).
        fusion_label:      Fusion architecture label ("MLP" or "Gated").
    """
    name: str
    label: str
    restaurant_groups: list[str]
    user_groups: list[str]
    extra_args: list[str] = field(default_factory=list)
    short_label: str = ""
    restaurant_desc: str = ""
    user_desc: str = ""
    fusion_label: str = "MLP"


# Reusable feature-group bundles ------------------------------------------------
REST_CONTENT = ["attributes", "categories", "price"]
REST_CONTEXT = ["temporal"]
REST_ALL = REST_CONTENT + REST_CONTEXT

USER_CONTEXT = ["day_of_week", "distance"]
USER_CONTENT = ["onboarding", "preferences"]
USER_ALL = USER_CONTEXT + USER_CONTENT


# ── The registry ──────────────────────────────────────────────────────────────

EXPERIMENTS: list[Experiment] = [
    # ── Full model ───────────────────────────────────────────────────────
    Experiment(
        name="full_model",
        label="Full model (all features)",
        restaurant_groups=REST_ALL,
        user_groups=USER_ALL,
        short_label="Full Model",
        restaurant_desc="Content + Context",
        user_desc="Content + Context",
    ),

    # ── Restaurant-side ablations ────────────────────────────────────────
    Experiment(
        name="no_temporal",
        label="No restaurant context (drop temporal checkin profile)",
        restaurant_groups=REST_CONTENT,
        user_groups=USER_ALL,
        short_label="No Temporal",
        restaurant_desc="Content Only",
        user_desc="Content + Context",
    ),
    Experiment(
        name="no_restaurant_content",
        label="No restaurant content (temporal only)",
        restaurant_groups=REST_CONTEXT,
        user_groups=USER_ALL,
        short_label="No Rest. Content",
        restaurant_desc="Context Only",
        user_desc="Content + Context",
    ),

    # ── User-side ablations ──────────────────────────────────────────────
    Experiment(
        name="no_user_context",
        label="No user context (drop dow/distance)",
        restaurant_groups=REST_ALL,
        user_groups=USER_CONTENT,
        short_label="No User Context",
        restaurant_desc="Content + Context",
        user_desc="Content Only",
    ),
    Experiment(
        name="no_user_content",
        label="No user content (drop preferences + onboarding)",
        restaurant_groups=REST_ALL,
        user_groups=USER_CONTEXT,
        short_label="No User Content",
        restaurant_desc="Content + Context",
        user_desc="Context Only",
    ),
    Experiment(
        name="no_onboarding",
        label="No onboarding (preferences only)",
        restaurant_groups=REST_ALL,
        user_groups=USER_CONTEXT + ["preferences"],
        short_label="No Onboarding",
        restaurant_desc="Content + Context",
        user_desc="Ctx + Prefs",
    ),
    Experiment(
        name="no_preferences",
        label="No preferences (onboarding only)",
        restaurant_groups=REST_ALL,
        user_groups=USER_CONTEXT + ["onboarding"],
        short_label="No Preferences",
        restaurant_desc="Content + Context",
        user_desc="Ctx + Onboard",
    ),

    # ── Cross-concern ablations ──────────────────────────────────────────
    Experiment(
        name="content_x_content",
        label="Content × Content (no context either side)",
        restaurant_groups=REST_CONTENT,
        user_groups=USER_CONTENT,
        short_label="Content × Content",
        restaurant_desc="Content Only",
        user_desc="Content Only",
    ),
    Experiment(
        name="context_x_context",
        label="Context × Context (no content either side — restaurant content still required)",
        restaurant_groups=REST_ALL,
        user_groups=USER_CONTEXT,
        short_label="Context × Context",
        restaurant_desc="Content + Context",
        user_desc="Context Only",
    ),

    # ── User checkin ablations ───────────────────────────────────────────
    Experiment(
        name="with_user_checkin",
        label="Full + user checkin time-of-visit",
        restaurant_groups=REST_ALL,
        user_groups=USER_ALL + ["user_checkin"],
        short_label="Full + Checkin",
        restaurant_desc="Content + Context",
        user_desc="Content + Context+",
    ),
    Experiment(
        name="checkin_replaces_dow",
        label="User checkin replaces day-of-week",
        restaurant_groups=REST_ALL,
        user_groups=["distance", "onboarding", "preferences", "user_checkin"],
        short_label="Checkin replaces DoW",
        restaurant_desc="Content + Context",
        user_desc="Content + Checkin",
    ),

    # ── Fusion ablations ─────────────────────────────────────────────────
    Experiment(
        name="gated_fusion",
        label="Gated fusion (full model, content-gated context mixing)",
        restaurant_groups=REST_ALL,
        user_groups=USER_ALL,
        extra_args=["--fusion", "gated"],
        short_label="Gated Fusion",
        restaurant_desc="Content + Context",
        user_desc="Content + Context",
        fusion_label="Gated",
    ),
    Experiment(
        name="gated_fusion_with_checkin",
        label="Gated fusion + user checkin time-of-visit",
        restaurant_groups=REST_ALL,
        user_groups=USER_ALL + ["user_checkin"],
        extra_args=["--fusion", "gated"],
        short_label="Gated + Checkin",
        restaurant_desc="Content + Context",
        user_desc="Content + Context+",
        fusion_label="Gated",
    ),
]

EXPERIMENT_MAP: dict[str, Experiment] = {exp.name: exp for exp in EXPERIMENTS}

# Convenience label lookups (name -> label).
EXPERIMENT_LABELS: dict[str, str] = {exp.name: exp.label for exp in EXPERIMENTS}
SHORT_LABELS: dict[str, str] = {exp.name: exp.short_label for exp in EXPERIMENTS}


# ── Reporting metadata ────────────────────────────────────────────────────────

# Logical ordering used by every reporting view (summary tables, plots).
DISPLAY_ORDER: list[str] = [
    # Core model
    "full_model",
    # Restaurant-side ablations
    "no_temporal", "no_restaurant_content",
    # User-side ablations
    "no_user_context", "no_user_content",
    # Cross-concern
    "content_x_content", "context_x_context",
    # User content breakdown
    "no_onboarding", "no_preferences",
    # User checkin
    "with_user_checkin", "checkin_replaces_dow",
    # Gated fusion
    "gated_fusion", "gated_fusion_with_checkin",
]


@dataclass(frozen=True)
class Comparison:
    """A curated A-vs-B comparison for the summary report."""
    title: str
    baseline: str   # experiment name treated as "A" (usually full_model)
    variant: str    # experiment name treated as "B" (the ablation)
    question: str


# The headline ablation contrasts that tell the study's story.
KEY_COMPARISONS: list[Comparison] = [
    Comparison(
        "Full vs No Restaurant Context", "full_model", "no_temporal",
        "Does temporal context on the restaurant side help?"),
    Comparison(
        "Full vs No Restaurant Content", "full_model", "no_restaurant_content",
        "How much do categories/price/attributes contribute?"),
    Comparison(
        "Full vs No User Context", "full_model", "no_user_context",
        "Does day-of-week/distance context help?"),
    Comparison(
        "Full vs No User Content", "full_model", "no_user_content",
        "Do user preference features help?"),
    Comparison(
        "Full vs Content × Content", "full_model", "content_x_content",
        "Does ANY context help vs pure content matching?"),
    Comparison(
        "Full vs Context × Context", "full_model", "context_x_context",
        "How far does context alone get you without content?"),
]

# Evaluation splits, shared so reporting views agree on order.
SPLITS: list[str] = ["warm", "cold_restaurant", "cold_user"]

# Canonical seeds for the multi-seed study. The runner trains every experiment
# once per seed and the summarizer aggregates (mean ± std) over exactly these,
# so training-noise variance is reported consistently everywhere. Override on
# the runner CLI (``--seeds``) for a cheaper/quicker sweep. The first seed is
# the "primary" one, used for seed-invariant quantities (test cases, baselines,
# full-corpus eval).
SEEDS: list[int] = [42, 43, 44, 45, 46]
PRIMARY_SEED: int = SEEDS[0]


# ── Integrity check ───────────────────────────────────────────────────────────
# Fail loudly at import time if the reporting metadata references an experiment
# that no longer exists — this is what makes drift impossible rather than silent.

def _validate() -> None:
    names = set(EXPERIMENT_MAP)
    dangling = [n for n in DISPLAY_ORDER if n not in names]
    if dangling:
        raise ValueError(f"DISPLAY_ORDER references unknown experiments: {dangling}")
    missing_from_order = names - set(DISPLAY_ORDER)
    if missing_from_order:
        raise ValueError(
            f"Experiments missing from DISPLAY_ORDER: {sorted(missing_from_order)}"
        )
    for cmp in KEY_COMPARISONS:
        for n in (cmp.baseline, cmp.variant):
            if n not in names:
                raise ValueError(
                    f"KEY_COMPARISONS entry {cmp.title!r} references unknown experiment {n!r}"
                )
    if not SEEDS:
        raise ValueError("SEEDS must be non-empty")
    if PRIMARY_SEED not in SEEDS:
        raise ValueError(f"PRIMARY_SEED {PRIMARY_SEED} must be one of SEEDS {SEEDS}")


_validate()
