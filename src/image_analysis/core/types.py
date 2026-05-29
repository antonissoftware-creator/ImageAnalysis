from __future__ import annotations

# Dataclass decorators for plain structured containers.
from dataclasses import dataclass, field
# Path type for source image references.
from pathlib import Path
# Type hints for maps and tuples.
from typing import Dict, List, Tuple

# Array type for image and feature tensors.
import numpy as np

# Alias for human-readable side labels.
Side = str
# Alias for piece identifier.
PieceId = int
# Alias for orientation in degrees.
Rotation = int


# Hold one puzzle instance and its hidden ground truth.
@dataclass
class PuzzleInstance:
    # Source image path used to generate this puzzle.
    image_path: Path
    # Shuffled piece images in observed order.
    pieces: List[np.ndarray]
    # Grid shape as (rows, cols).
    grid_shape: Tuple[int, int]
    # Piece ids in observed id space.
    piece_ids: List[int]
    # Ground-truth absolute positions by piece id.
    gt_positions: Dict[int, Tuple[int, int]]
    # Ground-truth rotation labels by piece id.
    gt_rotations: Dict[int, int]


# Hold all extracted descriptors used by compatibility stage.
@dataclass
class DescriptorBundle:
    # family -> (piece_id, rotation) -> descriptor vector.
    piece_level: Dict[str, Dict[Tuple[int, Rotation], np.ndarray]] = field(default_factory=dict)
    # family -> (piece_id, rotation, side) -> descriptor vector.
    side_level: Dict[str, Dict[Tuple[int, Rotation, Side], np.ndarray]] = field(default_factory=dict)


# Hold final solver output for one puzzle instance.
@dataclass
class ReconstructionResult:
    # Predicted absolute position by piece id.
    placement: Dict[int, Tuple[int, int]]
    # Predicted orientation by piece id.
    rotations: Dict[int, int]
    # Final objective value of selected solution.
    objective: float


# Hold quantitative evaluation metrics for one prediction.
@dataclass
class EvaluationReport:
    # Fraction of pieces with correct absolute placement.
    piece_accuracy: float
    # Fraction of true neighbors recovered in prediction.
    neighbor_accuracy: float
    # Fraction of pieces with correct orientation.
    rotation_accuracy: float
    # Extra scalar diagnostics.
    details: Dict[str, float] = field(default_factory=dict)
