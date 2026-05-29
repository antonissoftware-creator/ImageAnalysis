from __future__ import annotations

# Path handling for image discovery.
from pathlib import Path
# Type hints for image path lists.
from typing import List

# Numeric image arrays.
import numpy as np
# PIL backend for reading/writing/resizing images.
from PIL import Image


# Load image as RGB numpy array with optional long-edge resize.
def load_image(path: Path, resize_long_edge: int | None = None) -> np.ndarray:
    # Open image and normalize channel layout to RGB.
    image = Image.open(path).convert("RGB")
    # Resize only when long-edge cap is requested.
    if resize_long_edge is not None:
        # Read original image width/height.
        w, h = image.size
        # Compute scale factor against long edge.
        scale = resize_long_edge / max(w, h)
        # Downscale only when source is larger than cap.
        if scale < 1.0:
            image = image.resize((int(w * scale), int(h * scale)), Image.Resampling.BILINEAR)
    # Convert PIL image to numpy array.
    return np.asarray(image)


# Rotate image by multiples of 90 degrees.
def rotate_90_multiples(arr: np.ndarray, degrees: int) -> np.ndarray:
    # Convert degree value to np.rot90 step count.
    k = (degrees // 90) % 4
    # Return rotated view.
    return np.rot90(arr, k=k)


# Save RGB uint8 numpy array as image file.
def save_rgb(path: Path, arr: np.ndarray) -> None:
    # Persist array using PIL writer.
    Image.fromarray(arr.astype(np.uint8)).save(path)


# Recursively list supported image files.
def list_images(directory: Path) -> List[Path]:
    # Supported file extensions.
    exts = {".png", ".jpg", ".jpeg", ".bmp"}
    # Return sorted recursive image path list.
    return [p for p in sorted(directory.rglob("*")) if p.suffix.lower() in exts]
