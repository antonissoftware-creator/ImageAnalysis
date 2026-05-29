from __future__ import annotations

# Type hints for compatibility tensor structures.
from typing import Dict, Tuple

# Vector math backend.
import numpy as np

# Descriptor container from feature stage.
from image_analysis.core.types import DescriptorBundle


# Compute Euclidean distance between two vectors.
def _euclidean(a: np.ndarray, b: np.ndarray) -> float:
    # Return L2 norm as scalar distance.
    return float(np.linalg.norm(a - b))


# Compute cosine similarity between two vectors.
def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    # Compute product of vector norms for denominator.
    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    # Guard against division by zero for degenerate vectors.
    if denom == 0:
        return 0.0
    # Return normalized dot product in [-1, 1].
    return float(np.dot(a, b) / denom)


# Map distance value into similarity score.
def _sim_from_distance(dist: float, mode: str, lamb: float) -> float:
    # Use Gaussian kernel when requested.
    if mode == "gaussian":
        return float(np.exp(-lamb * (dist**2)))
    # Use reciprocal fallback similarity.
    return 1.0 / (1.0 + dist)


# Build directed compatibility tensor C[(i,ri,si,j,rj,sj)] across families.
def build_compatibility(
    bundle: DescriptorBundle,
    family_weights: Dict[str, float],
    similarity: str = "gaussian",
    distance: str = "euclidean",
    lamb: float = 1.0,
    side_weight: float = 0.7,
    tile_weight: float = 0.3,
) -> Dict[Tuple[int, int, str, int, int, str], float]:
    # Keep only families present in current descriptor bundle.
    families = [f for f in family_weights if f in bundle.side_level]
    # Read existing (piece,rotation) keys from first family.
    piece_rot_keys = next(iter(bundle.piece_level.values())).keys()
    # Recover unique piece ids.
    piece_ids = sorted({pid for pid, _ in piece_rot_keys})
    # Recover unique rotation states.
    rots = sorted({rot for _, rot in piece_rot_keys})
    # Define side ordering for nested loops.
    sides = ["top", "right", "bottom", "left"]

    # Initialize output compatibility dictionary.
    out: Dict[Tuple[int, int, str, int, int, str], float] = {}
    # Iterate first piece id.
    for i in piece_ids:
        # Iterate second piece id.
        for j in piece_ids:
            # Skip self-adjacency because a piece cannot neighbor itself.
            if i == j:
                continue
            # Iterate orientation of first piece.
            for ri in rots:
                # Iterate orientation of second piece.
                for rj in rots:
                    # Iterate side of first piece.
                    for si in sides:
                        # Iterate side of second piece.
                        for sj in sides:
                            # Initialize compatibility accumulator.
                            total = 0.0
                            # Fuse all enabled descriptor families.
                            for fam in families:
                                # Read family-level contribution weight.
                                alpha = float(family_weights.get(fam, 0.0))
                                # Read side-level vectors for the pair.
                                svec_i = bundle.side_level[fam][(i, ri, si)]
                                svec_j = bundle.side_level[fam][(j, rj, sj)]
                                # Read piece-level vectors for the pair.
                                tvec_i = bundle.piece_level[fam][(i, ri)]
                                tvec_j = bundle.piece_level[fam][(j, rj)]

                                # Use cosine similarity directly when requested.
                                if distance == "cosine":
                                    # Normalize cosine output from [-1,1] to [0,1].
                                    ssim = (_cosine(svec_i, svec_j) + 1.0) / 2.0
                                    # Normalize cosine output from [-1,1] to [0,1].
                                    tsim = (_cosine(tvec_i, tvec_j) + 1.0) / 2.0
                                else:
                                    # Compute side distance then map to similarity.
                                    ssim = _sim_from_distance(_euclidean(svec_i, svec_j), similarity, lamb)
                                    # Compute tile distance then map to similarity.
                                    tsim = _sim_from_distance(_euclidean(tvec_i, tvec_j), similarity, lamb)

                                # Add weighted side+tile contribution for this family.
                                total += alpha * (side_weight * ssim + tile_weight * tsim)
                            # Persist directed compatibility score.
                            out[(i, ri, si, j, rj, sj)] = total
    # Return full compatibility tensor.
    return out
