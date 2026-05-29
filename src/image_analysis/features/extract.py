from __future__ import annotations

# Type hints for descriptor maps.
from typing import Dict, List

# Numeric operations over images and feature vectors.
import numpy as np
# Deep feature extraction backend.
import torch
# Pre-trained CNN models.
import torchvision.models as models
# Image transforms for CNN input preparation.
import torchvision.transforms as T
# Convert RGB to grayscale for texture features.
from skimage.color import rgb2gray
# Co-occurrence matrix and derived texture properties.
from skimage.feature import graycomatrix, graycoprops

# Core data structures for puzzle and descriptor outputs.
from image_analysis.core.types import DescriptorBundle, PuzzleInstance
# Rotate pieces by multiples of 90 degrees.
from image_analysis.utils.image_ops import rotate_90_multiples

# Canonical side labels for each tile.
SIDES = ("top", "right", "bottom", "left")
# Allowed orientation states used by solver and descriptors.
ROTATIONS = (0, 90, 180, 270)


# Extract border strip from a rotated piece for a given side.
def _strip(piece: np.ndarray, side: str, w: int) -> np.ndarray:
    # Return top rows for top side descriptor support.
    if side == "top":
        return piece[:w, :, :]
    # Return bottom rows for bottom side descriptor support.
    if side == "bottom":
        return piece[-w:, :, :]
    # Return left columns for left side descriptor support.
    if side == "left":
        return piece[:, :w, :]
    # Return right columns for right side descriptor support.
    return piece[:, -w:, :]


# Build normalized per-channel histogram color descriptor.
def color_hist(piece: np.ndarray, bins: int = 16) -> np.ndarray:
    # Accumulate histogram for each RGB channel.
    hists = []
    # Iterate channel index in RGB order.
    for c in range(3):
        # Compute normalized histogram in [0, 255] range.
        hist, _ = np.histogram(piece[..., c], bins=bins, range=(0, 255), density=True)
        # Append one channel histogram.
        hists.append(hist)
    # Concatenate channels into one fixed-length descriptor.
    return np.concatenate(hists).astype(np.float32)


# Build texture descriptor from GLCM statistics.
def texture_descriptor(piece: np.ndarray) -> np.ndarray:
    # Convert piece to 8-bit grayscale for co-occurrence computation.
    gray = (rgb2gray(piece) * 255).astype(np.uint8)
    # Compute GLCM at multiple distances and orientations.
    glcm = graycomatrix(gray, [1, 2], [0, np.pi / 2], levels=256, symmetric=True, normed=True)
    # Collect mean scalar statistics as texture summary vector.
    props = [
        graycoprops(glcm, "contrast").mean(),
        graycoprops(glcm, "dissimilarity").mean(),
        graycoprops(glcm, "homogeneity").mean(),
        graycoprops(glcm, "energy").mean(),
        graycoprops(glcm, "correlation").mean(),
    ]
    # Convert statistics to float32 descriptor.
    return np.array(props, dtype=np.float32)


# Wrap pre-trained ResNet18 as fixed deep feature extractor.
class DeepExtractor:
    # Initialize model and preprocessing transforms once.
    def __init__(self) -> None:
        # Load official default ImageNet weights.
        weights = models.ResNet18_Weights.DEFAULT
        # Instantiate ResNet18 model.
        self.model = models.resnet18(weights=weights)
        # Set evaluation mode to disable training behavior.
        self.model.eval()
        # Keep convolutional body and remove final classifier.
        self.model = torch.nn.Sequential(*(list(self.model.children())[:-1]))
        # Build preprocessing pipeline expected by the model.
        self.tf = T.Compose(
            [
                T.ToTensor(),
                T.Resize((224, 224), antialias=True),
                T.Normalize(mean=weights.transforms().mean, std=weights.transforms().std),
            ]
        )

    # Return one pooled deep descriptor vector for a piece image.
    @torch.inference_mode()
    def __call__(self, piece: np.ndarray) -> np.ndarray:
        # Transform image and add batch dimension.
        x = self.tf(piece).unsqueeze(0)
        # Forward pass and flatten pooled output to 1D vector.
        feat = self.model(x).flatten(1).squeeze(0).cpu().numpy()
        # Return float32 vector for consistent downstream math.
        return feat.astype(np.float32)


# Extract piece-level and side-level descriptors for all pieces and rotations.
def extract_descriptors(
    puzzle: PuzzleInstance,
    border_width: int = 6,
    enabled_families: List[str] | None = None,
) -> DescriptorBundle:
    # Use default family set when not explicitly provided.
    if enabled_families is None:
        enabled_families = ["color", "texture", "deep"]

    # Pre-allocate piece-level descriptor maps by family.
    piece_level: Dict[str, Dict[tuple[int, int], np.ndarray]] = {fam: {} for fam in enabled_families}
    # Pre-allocate side-level descriptor maps by family.
    side_level: Dict[str, Dict[tuple[int, int, str], np.ndarray]] = {fam: {} for fam in enabled_families}

    # Build deep extractor only if deep family is enabled.
    deep = DeepExtractor() if "deep" in enabled_families else None

    # Iterate every shuffled puzzle piece by id.
    for pid, base_piece in enumerate(puzzle.pieces):
        # Iterate all allowed rotations so solver can score orientation states.
        for rot in ROTATIONS:
            # Rotate piece and force contiguous copy to avoid negative strides.
            piece = rotate_90_multiples(base_piece, rot).copy()

            # Compute and store piece-level color descriptor.
            if "color" in enabled_families:
                piece_level["color"][(pid, rot)] = color_hist(piece)
            # Compute and store piece-level texture descriptor.
            if "texture" in enabled_families:
                piece_level["texture"][(pid, rot)] = texture_descriptor(piece)
            # Compute and store piece-level deep descriptor.
            if deep is not None:
                piece_level["deep"][(pid, rot)] = deep(piece)

            # Iterate each side to compute border-strip descriptors.
            for s in SIDES:
                # Slice strip and copy to guarantee safe tensor conversion.
                strip = _strip(piece, s, border_width).copy()
                # Compute and store side-level color descriptor.
                if "color" in enabled_families:
                    side_level["color"][(pid, rot, s)] = color_hist(strip)
                # Compute and store side-level texture descriptor.
                if "texture" in enabled_families:
                    side_level["texture"][(pid, rot, s)] = texture_descriptor(strip)
                # Compute and store side-level deep descriptor.
                if deep is not None:
                    side_level["deep"][(pid, rot, s)] = deep(strip)

    # Return descriptor bundle consumed by compatibility stage.
    return DescriptorBundle(piece_level=piece_level, side_level=side_level)
