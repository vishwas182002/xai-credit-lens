"""Configuration loader for XAI Credit Lens."""

from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel


PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
CONFIGS_DIR = PROJECT_ROOT / "configs"


def load_yaml(config_name: str) -> dict[str, Any]:
    """Load a YAML config file from the configs directory."""
    config_path = CONFIGS_DIR / config_name
    if not config_path.exists():
        raise FileNotFoundError(f"Config not found: {config_path}")
    with open(config_path) as f:
        return yaml.safe_load(f)


def get_model_config() -> dict[str, Any]:
    return load_yaml("model_config.yaml")


def get_fairness_config() -> dict[str, Any]:
    return load_yaml("fairness_config.yaml")


def get_regulatory_config() -> dict[str, Any]:
    return load_yaml("regulatory_config.yaml")
