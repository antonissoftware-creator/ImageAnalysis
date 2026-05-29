from __future__ import annotations

from pathlib import Path

import numpy as np

from image_analysis.core.types import PuzzleInstance, ReconstructionResult
from image_analysis.eval.metrics import evaluate
from image_analysis.solver.reconstruct import local_search_reconstruct


def _toy_puzzle() -> PuzzleInstance:
    piece = np.zeros((8, 8, 3), dtype=np.uint8)
    pieces = [piece.copy() for _ in range(4)]
    return PuzzleInstance(
        image_path=Path("."),
        pieces=pieces,
        grid_shape=(2, 2),
        piece_ids=[0, 1, 2, 3],
        gt_positions={0: (0, 0), 1: (0, 1), 2: (1, 0), 3: (1, 1)},
        gt_rotations={0: 0, 1: 90, 2: 180, 3: 270},
    )


def test_evaluate_perfect_case() -> None:
    puzzle = _toy_puzzle()
    pred = ReconstructionResult(
        placement=puzzle.gt_positions.copy(),
        rotations=puzzle.gt_rotations.copy(),
        objective=1.0,
    )
    rep = evaluate(puzzle, pred)
    assert rep.piece_accuracy == 1.0
    assert rep.neighbor_accuracy == 1.0
    assert rep.rotation_accuracy == 1.0


def test_solver_constraints() -> None:
    puzzle = _toy_puzzle()
    comp = {}
    for i in puzzle.piece_ids:
        for j in puzzle.piece_ids:
            if i == j:
                continue
            for ri in (0, 90, 180, 270):
                for rj in (0, 90, 180, 270):
                    for si in ("top", "right", "bottom", "left"):
                        for sj in ("top", "right", "bottom", "left"):
                            comp[(i, ri, si, j, rj, sj)] = 0.0

    pred = local_search_reconstruct(puzzle, comp, restarts=2, iterations=20, seed=1)
    assert set(pred.placement.keys()) == set(puzzle.piece_ids)
    assert len(set(pred.placement.values())) == len(puzzle.piece_ids)
    assert all(r in (0, 90, 180, 270) for r in pred.rotations.values())
