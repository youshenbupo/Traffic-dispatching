"""Configuration utilities."""
import os
import yaml
import argparse
from typing import Dict, Any


def load_config(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f) or {}
    base_path = config.pop("_base_", None)
    if base_path is None:
        return config
    if not os.path.isabs(base_path):
        base_path = os.path.join(os.path.dirname(path), base_path)
    return merge_config(load_config(base_path), config)


def merge_config(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    """Recursively merge override dict into base dict."""
    result = base.copy()
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = merge_config(result[key], value)
        else:
            result[key] = value
    return result


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    parser.add_argument("--gpus", type=str, default=None, help="Comma-separated GPU ids (e.g. 2,3)")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--exp_name", type=str, default=None)
    return parser
