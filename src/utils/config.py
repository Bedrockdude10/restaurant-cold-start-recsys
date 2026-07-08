"""Configuration loading and validation."""

from pathlib import Path

import yaml


def load_config(config_path: str | Path) -> dict:
    """Load YAML config and return as dict."""
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    return config
