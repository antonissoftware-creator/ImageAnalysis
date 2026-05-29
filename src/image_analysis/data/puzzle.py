from __future__ import annotations

# Path handling for output folders.
from pathlib import Path
# Type hints for piece arrays and position maps.
from typing import Dict, List, Tuple

# Numeric randomization and array slicing.
import numpy as np

# Shared puzzle object type.
from image_analysis.core.types import PuzzleInstance
# Image read/rotate/write helpers.
from image_analysis.utils.image_ops import load_image, rotate_90_multiples, save_rgb
# Directory creation and metadata JSON writing helpers.
from image_analysis.utils.io import dump_json, ensure_dir


# Split full image into regular grid tiles and record original positions.
def split_into_tiles(image: np.ndarray, rows: int, cols: int) -> Tuple[List[np.ndarray], Dict[int, Tuple[int, int]]]:
    # Read image dimensions.
    h, w = image.shape[:2]
    # Compute tile height.
    tile_h = h // rows
    # Compute tile width.
    tile_w = w // cols
    # Initialize tile list.
    pieces: List[np.ndarray] = []
    # Initialize ground-truth position map.
    gt: Dict[int, Tuple[int, int]] = {}
    # Initialize linear piece id counter.
    idx = 0
    # Iterate grid rows.
    for r in range(rows):
        # Iterate grid columns.
        for c in range(cols):
            # Slice tile region and copy into standalone array.
            tile = image[r * tile_h:(r + 1) * tile_h, c * tile_w:(c + 1) * tile_w].copy()
            # Store tile in list.
            pieces.append(tile)
            # Store original row/col for this tile id.
            gt[idx] = (r, c)
            # Advance piece id.
            idx += 1
    # Return tile list and original placement map.
    return pieces, gt


# Build one shuffled puzzle with optional rotations and persist artifacts.
def build_puzzle(image_path: Path, out_dir: Path, rows: int, cols: int, angles: List[int], allow_rotations: bool, seed: int = 42) -> PuzzleInstance:
    # Initialize deterministic random generator.
    rng = np.random.default_rng(seed)
    # Load source image as RGB numpy array.
    image = load_image(image_path)
    # Split image to regular tiles and original positions.
    pieces, gt_positions = split_into_tiles(image, rows, cols)
    # Read number of pieces.
    n = len(pieces)

    # Keep original-piece rotation labels.
    rotations: Dict[int, int] = {}
    # Keep rotated piece arrays.
    rotated: List[np.ndarray] = []
    # Iterate each original piece.
    for i, p in enumerate(pieces):
        # Sample rotation if enabled, otherwise keep zero rotation.
        angle = int(rng.choice(angles)) if allow_rotations else 0
        # Record original-piece rotation.
        rotations[i] = angle
        # Apply rotation and store result.
        rotated.append(rotate_90_multiples(p, angle))

    # Sample permutation from original indices to shuffled order.
    perm = rng.permutation(n).tolist()
    # Build shuffled piece list according to permutation.
    shuffled = [rotated[i] for i in perm]
    # Assign shuffled piece ids as 0..n-1.
    shuffled_ids = list(range(n))

    # Ensure top-level puzzle directory exists.
    ensure_dir(out_dir)
    # Define pieces output directory.
    pieces_dir = out_dir / "pieces"
    # Ensure pieces directory exists.
    ensure_dir(pieces_dir)
    # Persist shuffled piece PNGs.
    for i, piece in enumerate(shuffled):
        save_rgb(pieces_dir / f"piece_{i:04d}.png", piece)

    # Build metadata dictionary for reproducibility.
    metadata = {
        "image_path": str(image_path),
        "grid_shape": [rows, cols],
        "permutation_from_original_to_shuffled_index": perm,
        "gt_positions": {str(k): [v[0], v[1]] for k, v in gt_positions.items()},
        "gt_rotations": {str(k): int(v) for k, v in rotations.items()},
    }
    # Persist metadata JSON.
    dump_json(out_dir / "metadata.json", metadata)

    # Build inverse mapping from shuffled index to original id.
    inv = {new_i: orig_i for new_i, orig_i in enumerate(perm)}
    # Map ground-truth positions into shuffled-index id space.
    mapped_gt_positions = {i: gt_positions[inv[i]] for i in range(n)}
    # Map ground-truth rotations into shuffled-index id space.
    mapped_gt_rotations = {i: rotations[inv[i]] for i in range(n)}

    # Return ready-to-use puzzle instance.
    return PuzzleInstance(
        image_path=image_path,
        pieces=shuffled,
        grid_shape=(rows, cols),
        piece_ids=shuffled_ids,
        gt_positions=mapped_gt_positions,
        gt_rotations=mapped_gt_rotations,
    )
