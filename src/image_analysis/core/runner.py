from __future__ import annotations

# JSON cloning for safe config mutation.
import json
# Dataclass container for run outputs.
from dataclasses import dataclass
# Timestamps for run metadata and folder naming.
from datetime import datetime
# Path handling for artifacts.
from pathlib import Path
# Stage-level timing measurements.
from time import perf_counter
# Type hints for config and results payloads.
from typing import Any, Dict, List

# Load puzzle instance from generated puzzle folder.
from image_analysis.data.loaders import load_puzzle_instance
# Compute quantitative metrics from prediction vs ground truth.
from image_analysis.eval.metrics import evaluate
# Render qualitative reconstruction images.
from image_analysis.eval.visuals import save_qualitative_outputs
# Build piece-level and side-level descriptors.
from image_analysis.features.extract import extract_descriptors
# Build directed compatibility tensor.
from image_analysis.modeling.compatibility import build_compatibility
# Run orientation-aware local-search solver.
from image_analysis.solver.reconstruct import local_search_reconstruct
# Shared JSON output and directory utilities.
from image_analysis.utils.io import dump_json, ensure_dir


# Structured return object for one completed run.
@dataclass
class DemoRunResult:
    # Folder where all run artifacts are persisted.
    run_dir: Path
    # Scalar metric values from evaluation stage.
    metrics: Dict[str, float]
    # Serialized prediction (placement + rotations + objective).
    prediction: Dict[str, Any]
    # Path to side-by-side GT vs reconstruction image.
    comparison_path: Path
    # Path to mismatch overlay image.
    mismatch_path: Path
    # Duration (seconds) per pipeline stage.
    stage_durations: Dict[str, float]
    # Effective run parameters (variant, solver settings, etc.).
    params: Dict[str, Any]


# Build one experiment variant config from base config.
def build_variant_config(cfg: Dict[str, Any], variant: str) -> Dict[str, Any]:
    # Deep-copy config so the base dictionary remains unchanged.
    c = json.loads(json.dumps(cfg))
    # Disable deep contribution for classical-only variant.
    if variant == "classical_only":
        c["features"]["deep"]["enabled"] = False
        c["adjacency"]["family_weights"]["deep"] = 0.0
    # Disable color/texture contribution for deep-only variant.
    elif variant == "deep_only":
        c["features"]["color"]["enabled"] = False
        c["features"]["texture"]["enabled"] = False
        c["adjacency"]["family_weights"]["color"] = 0.0
        c["adjacency"]["family_weights"]["texture"] = 0.0
    # Return variant-specific configuration.
    return c


# Discover prior runs and optionally filter by puzzle and variant.
def find_existing_runs(output_root: Path, puzzle_name: str | None = None, variant: str | None = None) -> List[Path]:
    # Return empty list when run root does not exist yet.
    if not output_root.exists():
        return []

    # Collect matching run directories.
    runs: List[Path] = []
    # Search for run metadata files recursively.
    for p in output_root.rglob("run_meta.json"):
        # Run directory is the parent of run_meta file.
        rdir = p.parent
        # Ignore malformed metadata files.
        try:
            meta = json.loads(p.read_text(encoding="utf-8"))
        except Exception:
            continue
        # Apply puzzle filter when requested.
        if puzzle_name is not None and meta.get("puzzle") != puzzle_name:
            continue
        # Apply variant filter when requested.
        if variant is not None and meta.get("variant") != variant:
            continue
        # Keep matching run directory.
        runs.append(rdir)
    # Return newest-first ordering by path name (timestamped folders).
    return sorted(runs, reverse=True)


# Internal helper: load puzzle and extract descriptor bundle.
def _extract_bundle(cfg: Dict[str, Any], puzzle_dir: Path, run_dir: Path):
    # Load shuffled pieces and hidden ground truth.
    puzzle = load_puzzle_instance(puzzle_dir)
    # Select descriptor families enabled in config.
    families = [k for k, v in cfg["features"].items() if isinstance(v, dict) and v.get("enabled", False)]
    # Build descriptor bundle across pieces/sides/rotations.
    bundle = extract_descriptors(
        puzzle,
        border_width=int(cfg["puzzle"]["border_width"]),
        enabled_families=families,
    )
    # Ensure run folder exists before writing metadata.
    ensure_dir(run_dir)
    # Persist selected feature settings for reproducibility.
    dump_json(
        run_dir / "features_meta.json",
        {"families": families, "border_width": int(cfg["puzzle"]["border_width"])},
    )
    # Return loaded puzzle and descriptors.
    return puzzle, bundle


# Internal helper: build compatibility tensor from descriptors and config.
def _build_comp(cfg: Dict[str, Any], bundle):
    # Normalize family weights to float.
    weights = {k: float(v) for k, v in cfg["adjacency"]["family_weights"].items()}
    # Compute full directed compatibility tensor.
    return build_compatibility(
        bundle,
        family_weights=weights,
        similarity=str(cfg["adjacency"].get("similarity", "gaussian")),
        distance=str(cfg["adjacency"].get("distance", "euclidean")),
        lamb=float(cfg["adjacency"].get("lambda", 1.0)),
    )


# Run one puzzle+variant end-to-end and persist all artifacts.
def run_single_puzzle_variant(
    cfg: Dict[str, Any],
    puzzle_dir: Path,
    variant: str,
    output_root: Path,
    solver_overrides: Dict[str, int] | None = None,
    timestamped: bool = True,
) -> DemoRunResult:
    # Capture start timestamp for metadata.
    started_at = datetime.utcnow().isoformat() + "Z"
    # Build local timestamp token for folder naming.
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Create timestamped folder for non-overwriting run history.
    if timestamped:
        run_dir = output_root / f"{stamp}__{puzzle_dir.name}__{variant}"
    else:
        # Use fixed folder layout for deterministic CLI outputs.
        run_dir = output_root / puzzle_dir.name / variant
    # Ensure run directory exists.
    ensure_dir(run_dir)

    # Initialize stage-duration dictionary.
    stage_durations: Dict[str, float] = {}

    # Time feature extraction stage.
    t0 = perf_counter()
    puzzle, bundle = _extract_bundle(cfg, puzzle_dir, run_dir)
    stage_durations["extract_features"] = perf_counter() - t0

    # Time compatibility construction stage.
    t1 = perf_counter()
    comp = _build_comp(cfg, bundle)
    stage_durations["build_compatibility"] = perf_counter() - t1

    # Time solver stage.
    t2 = perf_counter()
    # Resolve restart count from override or config.
    restarts = int((solver_overrides or {}).get("restarts", cfg["solver"].get("restarts", 8)))
    # Resolve iteration count from override or config.
    iterations = int((solver_overrides or {}).get("iterations", cfg["solver"].get("iterations", 1500)))
    # Run orientation-aware global reconstruction.
    pred = local_search_reconstruct(
        puzzle,
        comp,
        restarts=restarts,
        iterations=iterations,
        seed=int(cfg["puzzle"].get("seed", 42)),
    )
    stage_durations["solve"] = perf_counter() - t2

    # Time evaluation stage.
    t3 = perf_counter()
    report = evaluate(puzzle, pred)
    stage_durations["evaluate"] = perf_counter() - t3

    # Build metrics payload for reporting.
    metrics = {
        "piece_accuracy": report.piece_accuracy,
        "neighbor_accuracy": report.neighbor_accuracy,
        "rotation_accuracy": report.rotation_accuracy,
        "objective": pred.objective,
    }

    # Build prediction payload for reproducibility.
    prediction = {
        "placement": {str(k): list(v) for k, v in pred.placement.items()},
        "rotations": {str(k): int(v) for k, v in pred.rotations.items()},
        "objective": pred.objective,
    }

    # Persist metrics JSON.
    dump_json(run_dir / "metrics.json", metrics)
    # Persist prediction JSON.
    dump_json(run_dir / "prediction.json", prediction)

    # Time qualitative image generation stage.
    t4 = perf_counter()
    save_qualitative_outputs(puzzle, pred, run_dir)
    stage_durations["save_visuals"] = perf_counter() - t4

    # Persist effective parameter set for this run.
    params = {
        "puzzle": puzzle_dir.name,
        "variant": variant,
        "solver": {"restarts": restarts, "iterations": iterations},
        "features": {k: v for k, v in cfg["features"].items() if isinstance(v, dict)},
        "adjacency": cfg["adjacency"],
        "started_at": started_at,
        "finished_at": datetime.utcnow().isoformat() + "Z",
    }

    # Combine parameters, stage timings, and metrics into run metadata.
    run_meta = {
        **params,
        "stage_durations": stage_durations,
        "metrics": metrics,
    }
    # Persist run metadata JSON.
    dump_json(run_dir / "run_meta.json", run_meta)

    # Return typed run result for CLI/demo callers.
    return DemoRunResult(
        run_dir=run_dir,
        metrics=metrics,
        prediction=prediction,
        comparison_path=run_dir / "comparison.png",
        mismatch_path=run_dir / "mismatch_overlay.png",
        stage_durations=stage_durations,
        params=params,
    )
