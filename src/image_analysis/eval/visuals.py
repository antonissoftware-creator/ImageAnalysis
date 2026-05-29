from __future__ import annotations

# Path helpers for output image files.
from pathlib import Path
# Type hints for placement maps.
from typing import Dict, Tuple

# Plotting backend for saved figures.
import matplotlib.pyplot as plt
# Numeric array operations for canvas composition.
import numpy as np
# Image resizing for consistent cell shape.
from PIL import Image

# Shared puzzle and prediction types.
from image_analysis.core.types import PuzzleInstance, ReconstructionResult
# Rotation helper reused from puzzle stage.
from image_analysis.utils.image_ops import rotate_90_multiples
# Output directory creation helper.
from image_analysis.utils.io import ensure_dir


# Resize piece image to target tile shape when needed.
def _resize_to(arr: np.ndarray, h: int, w: int) -> np.ndarray:
    # Return as-is when target size already matches.
    if arr.shape[0] == h and arr.shape[1] == w:
        return arr
    # Convert array to PIL image for robust resizing.
    img = Image.fromarray(arr.astype(np.uint8))
    # Resize to requested width/height.
    img = img.resize((w, h), Image.Resampling.BILINEAR)
    # Convert resized image back to numpy.
    return np.asarray(img)


# Compose full reconstructed image canvas from placement and rotations.
def build_canvas(puzzle: PuzzleInstance, placement: Dict[int, Tuple[int, int]], rotations: Dict[int, int]) -> np.ndarray:
    # Read puzzle grid dimensions.
    rows, cols = puzzle.grid_shape
    # Read all piece heights for robust target size estimation.
    heights = [p.shape[0] for p in puzzle.pieces]
    # Read all piece widths for robust target size estimation.
    widths = [p.shape[1] for p in puzzle.pieces]
    # Use median piece size to reduce outlier effects.
    ph, pw = int(np.median(heights)), int(np.median(widths))

    # Allocate empty RGB canvas.
    canvas = np.zeros((rows * ph, cols * pw, 3), dtype=np.uint8)
    # Place each piece into its predicted or ground-truth location.
    for pid, (r, c) in placement.items():
        # Rotate piece to requested orientation.
        piece = rotate_90_multiples(puzzle.pieces[pid], int(rotations.get(pid, 0))).copy()
        # Resize piece back to canonical tile size if needed.
        piece = _resize_to(piece, ph, pw)
        # Paste piece into canvas cell.
        canvas[r * ph:(r + 1) * ph, c * pw:(c + 1) * pw] = piece
    # Return composed image canvas.
    return canvas


# Save qualitative images: comparison panel and mismatch overlay.
def save_qualitative_outputs(puzzle: PuzzleInstance, pred: ReconstructionResult, out_dir: Path) -> None:
    # Ensure output folder exists.
    ensure_dir(out_dir)
    # Build predicted reconstruction canvas.
    pred_img = build_canvas(puzzle, pred.placement, pred.rotations)
    # Build ground-truth reconstruction canvas.
    gt_img = build_canvas(puzzle, puzzle.gt_positions, puzzle.gt_rotations)

    # Start side-by-side comparison figure.
    plt.figure(figsize=(12, 6))
    # Select left subplot.
    plt.subplot(1, 2, 1)
    # Title left subplot.
    plt.title("Ground Truth")
    # Show ground-truth image.
    plt.imshow(gt_img)
    # Hide axis ticks.
    plt.axis("off")

    # Select right subplot.
    plt.subplot(1, 2, 2)
    # Title right subplot.
    plt.title("Reconstruction")
    # Show predicted image.
    plt.imshow(pred_img)
    # Hide axis ticks.
    plt.axis("off")
    # Improve layout spacing.
    plt.tight_layout()
    # Save comparison panel image.
    plt.savefig(out_dir / "comparison.png", dpi=150)
    # Close figure to free memory.
    plt.close()

    # Copy predicted image for mismatch annotations.
    overlay = pred_img.copy()
    # Compute tile height in composed canvas.
    ph = pred_img.shape[0] // puzzle.grid_shape[0]
    # Compute tile width in composed canvas.
    pw = pred_img.shape[1] // puzzle.grid_shape[1]
    # Iterate predicted piece locations.
    for pid, (r, c) in pred.placement.items():
        # Check if piece is correctly placed.
        correct = puzzle.gt_positions[pid] == (r, c)
        # Draw red border only for wrong pieces.
        if not correct:
            # Compute cell bounds in image coordinates.
            y0, y1 = r * ph, (r + 1) * ph
            x0, x1 = c * pw, (c + 1) * pw
            # Draw top border.
            overlay[y0:y0 + 3, x0:x1] = [255, 0, 0]
            # Draw bottom border.
            overlay[y1 - 3:y1, x0:x1] = [255, 0, 0]
            # Draw left border.
            overlay[y0:y1, x0:x0 + 3] = [255, 0, 0]
            # Draw right border.
            overlay[y0:y1, x1 - 3:x1] = [255, 0, 0]

    # Start mismatch overlay figure.
    plt.figure(figsize=(6, 6))
    # Title overlay figure.
    plt.title("Mismatched Piece Overlay")
    # Show overlay image.
    plt.imshow(overlay)
    # Hide axis ticks.
    plt.axis("off")
    # Improve layout spacing.
    plt.tight_layout()
    # Save mismatch overlay image.
    plt.savefig(out_dir / "mismatch_overlay.png", dpi=150)
    # Close figure to free memory.
    plt.close()
