import os
from typing import Dict, Any

import yaml

PROJECT_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), os.pardir, os.pardir)
)


def load_config() -> Dict[str, Any]:
    filepath = os.path.join(PROJECT_DIR, "config.yaml")
    with open(filepath, "r") as f:
        return yaml.safe_load(f)
