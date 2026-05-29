from __future__ import annotations

# Parse puzzle metadata JSON files.
import json
# Path utilities for puzzle folders and files.
from pathlib import Path
# Type hints for puzzle directory lists.
from typing import List

# Shared puzzle instance type.
from image_analysis.core.types import PuzzleInstance
# Image loader used to read piece PNG files.
from image_analysis.utils.image_ops import load_image


# Load one generated puzzle directory into an in-memory PuzzleInstance.
def load_puzzle_instance(puzzle_dir: Path) -> PuzzleInstance:
    # Resolve metadata path.
    meta = puzzle_dir / "metadata.json"
    # Parse metadata JSON.
    m = json.loads(meta.read_text(encoding="utf-8"))
    # Read grid dimensions.
    rows, cols = m["grid_shape"]
    # Load all shuffled piece images in deterministic order.
    pieces = [load_image(p) for p in sorted((puzzle_dir / "pieces").glob("piece_*.png"))]
    # Build and return strongly-typed puzzle object.
    return PuzzleInstance(
        image_path=Path(m["image_path"]),
        pieces=pieces,
        grid_shape=(rows, cols),
        piece_ids=list(range(len(pieces))),
        gt_positions={int(k): tuple(v) for k, v in m["gt_positions"].items()},
        gt_rotations={int(k): int(v) for k, v in m["gt_rotations"].items()},
    )


# Discover all puzzle directories that contain metadata.
def list_puzzle_dirs(root: Path) -> List[Path]:
    # Return sorted puzzle folders matching naming convention.
    return sorted([p for p in root.glob("puzzle_*") if (p / "metadata.json").exists()])
