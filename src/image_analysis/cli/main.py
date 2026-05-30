from __future__ import annotations

# Path handling for config and artifact directories.
from pathlib import Path
# Type hints for metric aggregation maps.
from typing import Dict, List

# Command-line framework.
import typer
# Console status rendering.
from rich import print

# Shared orchestration functions reused by CLI and demo.
from image_analysis.core.runner import (
    build_variant_config,
    run_single_puzzle_variant,
)
# Puzzle discovery utilities.
from image_analysis.data.loaders import list_puzzle_dirs
# Puzzle generation pipeline.
from image_analysis.data.puzzle import build_puzzle
# Config and JSON output helpers.
from image_analysis.utils.io import dump_json, ensure_dir, load_yaml

# Register root CLI app.
app = typer.Typer(help="Image jigsaw reconstruction CLI")


# Create required input/output folders.
@app.command("prepare-data")
def prepare_data(config: Path = Path("configs/default.yaml")) -> None:
    # Load YAML configuration.
    cfg = load_yaml(config)
    # Ensure raw input directory exists.
    ensure_dir(Path(cfg["data"]["input_dir"]))
    # Ensure generated puzzle directory exists.
    ensure_dir(Path(cfg["data"]["puzzle_dir"]))
    # Ensure evaluation output directory exists.
    ensure_dir(Path(cfg["evaluation"]["output_dir"]))
    # Print completion message.
    print("[green]Data directories prepared[/green]")


# Generate shuffled/rotated puzzles from input images.
@app.command("make-puzzles")
def make_puzzles(config: Path = Path("configs/default.yaml")) -> None:
    # Load YAML configuration.
    cfg = load_yaml(config)
    # Import image listing helper lazily.
    from image_analysis.utils.image_ops import list_images

    # Collect images and enforce max-image cap.
    images = list_images(Path(cfg["data"]["input_dir"]))[: int(cfg["data"]["max_images"])]
    # Abort when no input images are found.
    if not images:
        print("[red]No images found in input_dir[/red]")
        raise typer.Exit(1)

    # Build one puzzle per source image.
    for i, image_path in enumerate(images):
        # Create deterministic puzzle folder name.
        out = Path(cfg["data"]["puzzle_dir"]) / f"puzzle_{i:03d}"
        # Execute puzzle generation with configured parameters.
        build_puzzle(
            image_path=image_path,
            out_dir=out,
            rows=int(cfg["puzzle"]["grid_rows"]),
            cols=int(cfg["puzzle"]["grid_cols"]),
            angles=list(cfg["puzzle"]["rotation_angles"]),
            allow_rotations=bool(cfg["puzzle"]["allow_rotations"]),
            seed=int(cfg["puzzle"]["seed"]) + i,
        )
    # Print completion message.
    print(f"[green]Generated {len(images)} puzzles[/green]")


# Trigger lightweight feature-stage runs across all puzzles.
@app.command("extract-features")
def extract_features(config: Path = Path("configs/default.yaml")) -> None:
    # Load YAML configuration.
    cfg = load_yaml(config)
    # Discover generated puzzle directories.
    pdirs = list_puzzle_dirs(Path(cfg["data"]["puzzle_dir"]))
    # Abort when no puzzles exist.
    if not pdirs:
        print("[red]No puzzles found; run make-puzzles first[/red]")
        raise typer.Exit(1)

    # Define output folder for feature-stage artifacts.
    out = Path(cfg["evaluation"]["output_dir"]) / "features_only"
    # Run single-puzzle flow with tiny solver budget to materialize feature metadata.
    for p in pdirs:
        run_single_puzzle_variant(cfg, p, "combined", out, solver_overrides={"restarts": 1, "iterations": 1}, timestamped=False)
    # Print completion message.
    print(f"[green]Extracted features metadata for {len(pdirs)} puzzles[/green]")


# Trigger compatibility-stage preview run on first puzzle.
@app.command("build-compatibility")
def build_compat(config: Path = Path("configs/default.yaml")) -> None:
    # Load YAML configuration.
    cfg = load_yaml(config)
    # Discover generated puzzle directories.
    pdirs = list_puzzle_dirs(Path(cfg["data"]["puzzle_dir"]))
    # Abort when no puzzles exist.
    if not pdirs:
        print("[red]No puzzles found; run make-puzzles first[/red]")
        raise typer.Exit(1)

    # Define output folder for compatibility preview.
    out = Path(cfg["evaluation"]["output_dir"]) / "compat_preview"
    # Run first puzzle with minimal solver settings.
    res = run_single_puzzle_variant(cfg, pdirs[0], "combined", out, solver_overrides={"restarts": 1, "iterations": 1}, timestamped=False)
    # Persist pointer to produced run directory.
    dump_json(out / "compat_stats.json", {"run_dir": str(res.run_dir)})
    # Print completion message.
    print("[green]Compatibility built[/green]")


# Run full reconstruction on the first puzzle with current config.
@app.command("reconstruct")
def reconstruct(config: Path = Path("configs/default.yaml")) -> None:
    # Load YAML configuration.
    cfg = load_yaml(config)
    # Discover generated puzzle directories.
    pdirs = list_puzzle_dirs(Path(cfg["data"]["puzzle_dir"]))
    # Abort when no puzzles exist.
    if not pdirs:
        print("[red]No puzzles found; run make-puzzles first[/red]")
        raise typer.Exit(1)

    # Define evaluation output root.
    out = Path(cfg["evaluation"]["output_dir"])
    # Run combined variant on first puzzle.
    res = run_single_puzzle_variant(cfg, pdirs[0], "combined", out, timestamped=False)
    # Print completion message and metrics.
    print("[green]Reconstruction completed[/green]")
    print(res.metrics)


# Alias evaluate command to reconstruction flow.
@app.command("evaluate")
def evaluate_cmd(config: Path = Path("configs/default.yaml")) -> None:
    # Reuse reconstruction command behavior.
    reconstruct(config)


# Run all required ablation variants across all puzzles.
@app.command("run-experiments")
def run_experiments(config: Path = Path("configs/default.yaml")) -> None:
    # Load YAML configuration.
    cfg = load_yaml(config)
    # Discover generated puzzle directories.
    pdirs = list_puzzle_dirs(Path(cfg["data"]["puzzle_dir"]))
    # Abort when no puzzles exist.
    if not pdirs:
        print("[red]No puzzles found; run make-puzzles first[/red]")
        raise typer.Exit(1)

    # Define mandatory experiment variants.
    variants = ["combined", "classical_only", "deep_only"]
    # Resolve output directory for experiment artifacts.
    out_dir = Path(cfg["evaluation"]["output_dir"])
    # Ensure output directory exists.
    ensure_dir(out_dir)

    # Initialize summary metrics map.
    summary: Dict[str, Dict[str, float]] = {}
    # Iterate each variant.
    for variant in variants:
        # Build variant-specific config.
        vcfg = build_variant_config(cfg, variant)
        # Collect per-puzzle metric dictionaries.
        vals: List[Dict[str, float]] = []
        # Run this variant across all puzzles.
        for p in pdirs:
            rep = run_single_puzzle_variant(vcfg, p, variant, out_dir, timestamped=False)
            vals.append(rep.metrics)

        # Compute mean metrics across puzzles for this variant.
        summary[variant] = {
            "piece_accuracy": float(sum(v["piece_accuracy"] for v in vals) / len(vals)),
            "neighbor_accuracy": float(sum(v["neighbor_accuracy"] for v in vals) / len(vals)),
            "rotation_accuracy": float(sum(v["rotation_accuracy"] for v in vals) / len(vals)),
        }

    # Persist summary metrics JSON.
    dump_json(out_dir / "summary_metrics.json", summary)
    # Print completion message and summary.
    print("[green]Experiment matrix completed[/green]")
    print(summary)


# Print Streamlit launch command for D3 app.
@app.command("demo")
def demo() -> None:
    # Keep demo command as explicit run hint.
    print("Run: streamlit run src/image_analysis/demo/app.py")
