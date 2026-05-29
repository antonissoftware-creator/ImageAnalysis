from __future__ import annotations

# JSON serialization for artifact outputs.
import json
# Path handling for files/directories.
from pathlib import Path
# Type hints for generic config dictionaries.
from typing import Any, Dict

# YAML parser for project configs.
import yaml


# Create directory tree when missing.
def ensure_dir(path: Path) -> None:
    # Build directory including all parents.
    path.mkdir(parents=True, exist_ok=True)


# Load YAML file into dictionary.
def load_yaml(path: Path) -> Dict[str, Any]:
    # Open YAML file with utf-8 encoding.
    with path.open("r", encoding="utf-8") as f:
        # Parse YAML into Python dict.
        return yaml.safe_load(f)


# Save dictionary as pretty JSON file.
def dump_json(path: Path, data: Dict[str, Any]) -> None:
    # Ensure parent directory exists before write.
    ensure_dir(path.parent)
    # Open destination JSON file with utf-8 encoding.
    with path.open("w", encoding="utf-8") as f:
        # Write indented JSON for human readability.
        json.dump(data, f, indent=2)
