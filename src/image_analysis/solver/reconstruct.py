from __future__ import annotations

# Lightweight state container for local-search configuration.
from dataclasses import dataclass
# Type hints for placement and score maps.
from typing import Dict, List, Tuple

# Randomized search and grid handling.
import numpy as np

# Puzzle input and reconstruction output types.
from image_analysis.core.types import PuzzleInstance, ReconstructionResult

# Allowed orientation states for each piece.
ROTATIONS = (0, 90, 180, 270)


# Hold one candidate solution (piece grid + per-piece rotation).
@dataclass
class Config:
    # Grid cell content is piece id.
    grid: np.ndarray
    # Rotation assignment per piece id.
    rot: Dict[int, int]


# Enumerate all horizontal/vertical neighbor edges and touching sides.
def _neighbors(rows: int, cols: int) -> List[Tuple[Tuple[int, int], Tuple[int, int], str, str]]:
    # Initialize edge list.
    edges: List[Tuple[Tuple[int, int], Tuple[int, int], str, str]] = []
    # Iterate all grid rows.
    for r in range(rows):
        # Iterate all grid columns.
        for c in range(cols):
            # Add right neighbor edge when it exists.
            if c + 1 < cols:
                edges.append(((r, c), (r, c + 1), "right", "left"))
            # Add bottom neighbor edge when it exists.
            if r + 1 < rows:
                edges.append(((r, c), (r + 1, c), "bottom", "top"))
    # Return all undirected grid edges with directed side labels.
    return edges


# Score one full configuration by summing compatibility on all neighbor edges.
def _score(config: Config, comp: Dict[Tuple[int, int, str, int, int, str], float], edges) -> float:
    # Initialize total objective value.
    total = 0.0
    # Iterate every adjacent grid pair.
    for (r1, c1), (r2, c2), s1, s2 in edges:
        # Read piece id in first cell.
        i = int(config.grid[r1, c1])
        # Read piece id in second cell.
        j = int(config.grid[r2, c2])
        # Read first piece rotation.
        ri = int(config.rot[i])
        # Read second piece rotation.
        rj = int(config.rot[j])
        # Add compatibility term for touching sides under current rotations.
        total += comp.get((i, ri, s1, j, rj, s2), 0.0)
    # Return objective value.
    return total


# Create random initial configuration for one restart.
def _initial_config(puzzle: PuzzleInstance, rng: np.random.Generator) -> Config:
    # Read puzzle grid size.
    rows, cols = puzzle.grid_shape
    # Convert piece ids to array for shuffling.
    ids = np.array(puzzle.piece_ids, dtype=int)
    # Randomly shuffle piece ordering.
    rng.shuffle(ids)
    # Fill grid row-major with shuffled ids.
    grid = ids.reshape(rows, cols).copy()
    # Randomly initialize piece rotations.
    rot = {pid: int(rng.choice(ROTATIONS)) for pid in puzzle.piece_ids}
    # Return random candidate solution.
    return Config(grid=grid, rot=rot)


# Reconstruct puzzle with multi-restart local search over swap/rotation moves.
def local_search_reconstruct(
    puzzle: PuzzleInstance,
    comp: Dict[Tuple[int, int, str, int, int, str], float],
    restarts: int = 8,
    iterations: int = 1500,
    seed: int = 42,
) -> ReconstructionResult:
    # Read puzzle grid shape.
    rows, cols = puzzle.grid_shape
    # Seed random generator for reproducibility.
    rng = np.random.default_rng(seed)
    # Precompute all grid neighbor edges once.
    edges = _neighbors(rows, cols)

    # Track best configuration across restarts.
    best_cfg = None
    # Track best objective across restarts.
    best_val = float("-inf")

    # Start each random restart.
    for _ in range(restarts):
        # Sample initial candidate.
        cfg = _initial_config(puzzle, rng)
        # Score initial candidate.
        cur = _score(cfg, comp, edges)

        # Run local-search iterations for this restart.
        for _it in range(iterations):
            # Track whether this iteration improved objective.
            improved = False

            # Pick one random piece for rotation move.
            pid = int(rng.choice(puzzle.piece_ids))
            # Store current rotation for rollback.
            oldr = cfg.rot[pid]
            # Enumerate alternate rotations.
            candidates = [r for r in ROTATIONS if r != oldr]
            # Sample one alternate rotation.
            newr = int(rng.choice(candidates))
            # Apply proposed rotation.
            cfg.rot[pid] = newr
            # Re-score candidate after rotation move.
            s = _score(cfg, comp, edges)
            # Keep move only if objective improves.
            if s > cur:
                cur = s
                improved = True
            else:
                # Revert rejected rotation.
                cfg.rot[pid] = oldr

            # Sample first cell for swap move.
            r1, c1 = int(rng.integers(0, rows)), int(rng.integers(0, cols))
            # Sample second cell for swap move.
            r2, c2 = int(rng.integers(0, rows)), int(rng.integers(0, cols))
            # Only evaluate swap when cells are distinct.
            if (r1, c1) != (r2, c2):
                # Apply swap proposal.
                cfg.grid[r1, c1], cfg.grid[r2, c2] = cfg.grid[r2, c2], cfg.grid[r1, c1]
                # Re-score candidate after swap.
                s2 = _score(cfg, comp, edges)
                # Keep swap only if objective improves.
                if s2 > cur:
                    cur = s2
                    improved = True
                else:
                    # Revert rejected swap.
                    cfg.grid[r1, c1], cfg.grid[r2, c2] = cfg.grid[r2, c2], cfg.grid[r1, c1]

            # Apply occasional random rotation to escape local minima.
            if not improved and rng.random() < 0.05:
                pid2 = int(rng.choice(puzzle.piece_ids))
                cfg.rot[pid2] = int(rng.choice(ROTATIONS))

        # Score final state of this restart.
        final = _score(cfg, comp, edges)
        # Keep global best across restarts.
        if final > best_val:
            best_val = final
            best_cfg = Config(grid=cfg.grid.copy(), rot=cfg.rot.copy())

    # Ensure at least one restart produced a candidate.
    assert best_cfg is not None
    # Convert best grid representation to piece->position mapping.
    placement: Dict[int, Tuple[int, int]] = {}
    # Iterate final grid rows.
    for r in range(rows):
        # Iterate final grid columns.
        for c in range(cols):
            # Save position of piece at current cell.
            placement[int(best_cfg.grid[r, c])] = (r, c)

    # Return final reconstruction object.
    return ReconstructionResult(placement=placement, rotations=best_cfg.rot, objective=best_val)
