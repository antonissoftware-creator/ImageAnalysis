from __future__ import annotations

# Parse and clone configs for experiment variants.
import json
# Handle filesystem paths.
from pathlib import Path
# Type hints for readability and safer refactors.
from typing import Any, Dict, List

# Build command-line interface.
import typer
# Print readable console status messages.
from rich import print

# Load generated puzzle folders and puzzle instances.
from image_analysis.data.loaders import list_puzzle_dirs, load_puzzle_instance
# Create shuffled/rotated puzzles from raw images.
from image_analysis.data.puzzle import build_puzzle
# Compute quantitative reconstruction metrics.
from image_analysis.eval.metrics import evaluate
# Export side-by-side and mismatch visuals.
from image_analysis.eval.visuals import save_qualitative_outputs
# Extract piece-level and side-level descriptors.
from image_analysis.features.extract import extract_descriptors
# Build directed compatibility scores between candidate neighbors.
from image_analysis.modeling.compatibility import build_compatibility
# Run orientation-aware local-search reconstruction.
from image_analysis.solver.reconstruct import local_search_reconstruct
# Read config and write JSON artifacts.
from image_analysis.utils.io import dump_json, ensure_dir, load_yaml

# Register root CLI app.
app = typer.Typer(help="Image jigsaw reconstruction CLI")


# Build one descriptor bundle for one puzzle and persist minimal run metadata.
def _extract_bundle(cfg: Dict[str, Any], puzzle_dir: Path, run_dir: Path):
    # Load puzzle pieces and hidden ground truth from metadata.
    puzzle = load_puzzle_instance(puzzle_dir)
    # Keep only descriptor families explicitly enabled in config.
    families = [k for k, v in cfg["features"].items() if isinstance(v, dict) and v.get("enabled", False)]
    # Extract all requested piece/side descriptors across allowed rotations.
    bundle = extract_descriptors(puzzle, border_width=int(cfg["puzzle"]["border_width"]), enabled_families=families)
    # Ensure output folder exists before writing metadata.
    ensure_dir(run_dir)
    # Save human-readable marker for what feature families were used.
    dump_json(run_dir / "features_meta.json", {"families": families, "border_width": int(cfg["puzzle"]["border_width"])})
    # Return in-memory objects for downstream solver pipeline.
    return puzzle, bundle


# Build compatibility tensor from descriptor bundle and adjacency config.
def _build_comp(cfg: Dict[str, Any], bundle):
    # Normalize weights to float for mathematical operations.
    weights = {k: float(v) for k, v in cfg["adjacency"]["family_weights"].items()}
    # Compute directed pairwise compatibility for all piece/rotation/side combinations.
    return build_compatibility(
        bundle,
        family_weights=weights,
        similarity=str(cfg["adjacency"].get("similarity", "gaussian")),
        distance=str(cfg["adjacency"].get("distance", "euclidean")),
        lamb=float(cfg["adjacency"].get("lambda", 1.0)),
    )


# Execute one full reconstruction run for one puzzle and one experiment variant.
def _run_single(cfg: Dict[str, Any], puzzle_dir: Path, out_dir: Path, exp_name: str):
    # Define per-puzzle, per-variant run folder.
    run_dir = out_dir / puzzle_dir.name / exp_name
    # Ensure folder exists.
    ensure_dir(run_dir)

    # Extract descriptors for this puzzle.
    puzzle, bundle = _extract_bundle(cfg, puzzle_dir, run_dir)
    # Build compatibility scores used by the solver.
    comp = _build_comp(cfg, bundle)

    # Run orientation-aware local-search reconstruction.
    pred = local_search_reconstruct(
        puzzle,
        comp,
        restarts=int(cfg["solver"].get("restarts", 8)),
        iterations=int(cfg["solver"].get("iterations", 1500)),
        seed=int(cfg["puzzle"].get("seed", 42)),
    )
    # Evaluate prediction against hidden ground truth.
    report = evaluate(puzzle, pred)

    # Save metric artifact for report tables.
    dump_json(
        run_dir / "metrics.json",
        {
            "piece_accuracy": report.piece_accuracy,
            "neighbor_accuracy": report.neighbor_accuracy,
            "rotation_accuracy": report.rotation_accuracy,
            "objective": pred.objective,
        },
    )
    # Save raw predicted placement/orientation for reproducibility.
    dump_json(
        run_dir / "prediction.json",
        {
            "placement": {str(k): list(v) for k, v in pred.placement.items()},
            "rotations": {str(k): int(v) for k, v in pred.rotations.items()},
            "objective": pred.objective,
        },
    )
    # Save qualitative visualization outputs for analysis/presentation.
    save_qualitative_outputs(puzzle, pred, run_dir)
    # Return evaluation report for aggregate summaries.
    return report


# Create config variant for ablation experiments.
def _variant_cfg(cfg: Dict[str, Any], name: str) -> Dict[str, Any]:
    # Deep-copy config via JSON round-trip to avoid mutating original.
    c = json.loads(json.dumps(cfg))
    # Disable deep features for classical-only run.
    if name == "classical_only":
        c["features"]["deep"]["enabled"] = False
        c["adjacency"]["family_weights"]["deep"] = 0.0
    # Disable classical feature families for deep-only run.
    elif name == "deep_only":
        c["features"]["color"]["enabled"] = False
        c["features"]["texture"]["enabled"] = False
        c["adjacency"]["family_weights"]["color"] = 0.0
        c["adjacency"]["family_weights"]["texture"] = 0.0
    # Return variant config.
    return c


# Prepare expected input/output directories.
@app.command("prepare-data")
def prepare_data(config: Path = Path("configs/default.yaml")) -> None:
    # Load configuration from YAML.
    cfg = load_yaml(config)
    # Create raw image input directory.
    ensure_dir(Path(cfg["data"]["input_dir"]))
    # Create generated puzzle output directory.
    ensure_dir(Path(cfg["data"]["puzzle_dir"]))
    # Create experiment results directory.
    ensure_dir(Path(cfg["evaluation"]["output_dir"]))
    # Confirm success in terminal.
    print("[green]Data directories prepared[/green]")


# Generate shuffled/rotated puzzles from raw images.
@app.command("make-puzzles")
def make_puzzles(config: Path = Path("configs/default.yaml")) -> None:
    # Load configuration from YAML.
    cfg = load_yaml(config)
    # Import image listing helper lazily.
    from image_analysis.utils.image_ops import list_images

    # Read input images and apply max-image cap.
    images = list_images(Path(cfg["data"]["input_dir"]))[: int(cfg["data"]["max_images"])]
    # Stop early when input directory is empty.
    if not images:
        print("[red]No images found in input_dir[/red]")
        raise typer.Exit(1)

    # Build one puzzle folder per source image.
    for i, image_path in enumerate(images):
        # Define deterministic puzzle folder name.
        out = Path(cfg["data"]["puzzle_dir"]) / f"puzzle_{i:03d}"
        # Run puzzle generation with configured grid/rotation settings.
        build_puzzle(
            image_path=image_path,
            out_dir=out,
            rows=int(cfg["puzzle"]["grid_rows"]),
            cols=int(cfg["puzzle"]["grid_cols"]),
            angles=list(cfg["puzzle"]["rotation_angles"]),
            allow_rotations=bool(cfg["puzzle"]["allow_rotations"]),
            seed=int(cfg["puzzle"]["seed"]) + i,
        )
    # Confirm number of generated puzzles.
    print(f"[green]Generated {len(images)} puzzles[/green]")


# Run feature extraction stage across all puzzle folders.
@app.command("extract-features")
def extract_features(config: Path = Path("configs/default.yaml")) -> None:
    # Load configuration.
    cfg = load_yaml(config)
    # Define staging directory for feature metadata.
    out = Path(cfg["evaluation"]["output_dir"]) / "features_only"
    # Discover generated puzzle folders.
    pdirs = list_puzzle_dirs(Path(cfg["data"]["puzzle_dir"]))
    # Guard against missing generated puzzles.
    if not pdirs:
        print("[red]No puzzles found; run make-puzzles first[/red]")
        raise typer.Exit(1)
    # Extract bundle for each puzzle.
    for p in pdirs:
        _extract_bundle(cfg, p, out / p.name)
    # Print completion summary.
    print(f"[green]Extracted features metadata for {len(pdirs)} puzzles[/green]")


# Run compatibility-building stage preview and persist tensor size stats.
@app.command("build-compatibility")
def build_compat(config: Path = Path("configs/default.yaml")) -> None:
    # Load configuration.
    cfg = load_yaml(config)
    # Discover generated puzzle folders.
    pdirs = list_puzzle_dirs(Path(cfg["data"]["puzzle_dir"]))
    # Guard against missing generated puzzles.
    if not pdirs:
        print("[red]No puzzles found; run make-puzzles first[/red]")
        raise typer.Exit(1)
    # Use first puzzle as a quick compatibility preview.
    p = pdirs[0]
    # Extract descriptors for preview run.
    _, bundle = _extract_bundle(cfg, p, Path(cfg["evaluation"]["output_dir"]) / "compat_preview")
    # Build compatibility tensor.
    comp = _build_comp(cfg, bundle)
    # Save number of compatibility entries.
    dump_json(Path(cfg["evaluation"]["output_dir"]) / "compat_preview" / "compat_stats.json", {"num_entries": len(comp)})
    # Confirm completion.
    print("[green]Compatibility built[/green]")


# Run one full reconstruction on first puzzle.
@app.command("reconstruct")
def reconstruct(config: Path = Path("configs/default.yaml")) -> None:
    # Load configuration.
    cfg = load_yaml(config)
    # Discover generated puzzle folders.
    pdirs = list_puzzle_dirs(Path(cfg["data"]["puzzle_dir"]))
    # Guard against missing generated puzzles.
    if not pdirs:
        print("[red]No puzzles found; run make-puzzles first[/red]")
        raise typer.Exit(1)
    # Execute single combined run.
    report = _run_single(cfg, pdirs[0], Path(cfg["evaluation"]["output_dir"]), "combined")
    # Print completion and metrics.
    print("[green]Reconstruction completed[/green]")
    print(report)


# Alias evaluate command to full reconstruction run.
@app.command("evaluate")
def evaluate_cmd(config: Path = Path("configs/default.yaml")) -> None:
    # Reuse reconstruction flow to compute metrics.
    reconstruct(config)


# Run full experiment matrix across all puzzles and feature variants.
@app.command("run-experiments")
def run_experiments(config: Path = Path("configs/default.yaml")) -> None:
    # Load configuration.
    cfg = load_yaml(config)
    # Discover generated puzzle folders.
    pdirs = list_puzzle_dirs(Path(cfg["data"]["puzzle_dir"]))
    # Guard against missing generated puzzles.
    if not pdirs:
        print("[red]No puzzles found; run make-puzzles first[/red]")
        raise typer.Exit(1)

    # Define required ablation variants.
    variants = ["combined", "classical_only", "deep_only"]
    # Resolve output root.
    out_dir = Path(cfg["evaluation"]["output_dir"])
    # Ensure output root exists.
    ensure_dir(out_dir)

    # Aggregate metrics per variant.
    summary: Dict[str, Dict[str, float]] = {}
    # Iterate each feature variant.
    for variant in variants:
        # Build variant-specific config.
        vcfg = _variant_cfg(cfg, variant)
        # Hold per-puzzle metrics for this variant.
        vals: List[Dict[str, float]] = []
        # Run variant on every puzzle.
        for p in pdirs:
            rep = _run_single(vcfg, p, out_dir, variant)
            vals.append(
                {
                    "piece_accuracy": rep.piece_accuracy,
                    "neighbor_accuracy": rep.neighbor_accuracy,
                    "rotation_accuracy": rep.rotation_accuracy,
                }
            )
        # Compute mean metrics for this variant.
        summary[variant] = {
            "piece_accuracy": float(sum(v["piece_accuracy"] for v in vals) / len(vals)),
            "neighbor_accuracy": float(sum(v["neighbor_accuracy"] for v in vals) / len(vals)),
            "rotation_accuracy": float(sum(v["rotation_accuracy"] for v in vals) / len(vals)),
        }

    # Save experiment summary JSON.
    dump_json(out_dir / "summary_metrics.json", summary)
    # Print completion status and summary.
    print("[green]Experiment matrix completed[/green]")
    print(summary)


# Print demo launch command.
@app.command("demo")
def demo() -> None:
    # Keep demo command as explicit streamlit runner hint.
    print("Run: streamlit run src/image_analysis/demo/app.py")
