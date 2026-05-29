from __future__ import annotations

# Type hints for mapping and set operations.
from typing import Dict, Set, Tuple

# Shared evaluation and puzzle types.
from image_analysis.core.types import EvaluationReport, PuzzleInstance, ReconstructionResult


# Convert absolute placements to unordered neighboring piece pairs.
def _neighbors_from_positions(pos: Dict[int, Tuple[int, int]]) -> Set[Tuple[int, int]]:
    # Invert mapping to cell->piece for neighborhood scanning.
    inverse = {v: k for k, v in pos.items()}
    # Initialize neighbor-pair set.
    out: Set[Tuple[int, int]] = set()
    # Iterate each occupied cell.
    for (r, c), pid in inverse.items():
        # Check only right/down to avoid duplicate pair counting.
        for nr, nc in ((r + 1, c), (r, c + 1)):
            # Keep pair only when neighbor cell exists.
            if (nr, nc) in inverse:
                # Read pair piece ids.
                a, b = pid, inverse[(nr, nc)]
                # Store sorted tuple so pair is order-invariant.
                out.add(tuple(sorted((a, b))))
    # Return neighboring pair set.
    return out


# Compute required assignment metrics for one reconstruction.
def evaluate(puzzle: PuzzleInstance, pred: ReconstructionResult) -> EvaluationReport:
    # Read total number of pieces.
    n = len(puzzle.piece_ids)
    # Count pieces placed in correct absolute positions.
    piece_ok = sum(int(pred.placement.get(pid) == puzzle.gt_positions[pid]) for pid in puzzle.piece_ids)
    # Count pieces with correct estimated orientation.
    rot_ok = sum(int(pred.rotations.get(pid, 0) == puzzle.gt_rotations[pid]) for pid in puzzle.piece_ids)

    # Build ground-truth neighbor pair set.
    gt_neigh = _neighbors_from_positions(puzzle.gt_positions)
    # Build predicted neighbor pair set.
    pr_neigh = _neighbors_from_positions(pred.placement)
    # Count correctly recovered neighbor pairs.
    inter = len(gt_neigh.intersection(pr_neigh))
    # Compute neighbor accuracy as recall over true neighbor pairs.
    neigh_acc = inter / max(len(gt_neigh), 1)

    # Return strongly-typed metric report.
    return EvaluationReport(
        piece_accuracy=piece_ok / n,
        neighbor_accuracy=neigh_acc,
        rotation_accuracy=rot_ok / n,
        details={"n_pieces": float(n)},
    )
