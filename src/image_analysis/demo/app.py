from __future__ import annotations

# Parse cached JSON artifacts.
import json
# Path handling for config, puzzles, and run folders.
from pathlib import Path

# Streamlit UI framework.
import streamlit as st

# Shared runner functions for variant config, execution, and history lookup.
from image_analysis.core.runner import (
    build_variant_config,
    find_existing_runs,
    run_single_puzzle_variant,
)
# Puzzle directory discovery helper.
from image_analysis.data.loaders import list_puzzle_dirs
# YAML config loader.
from image_analysis.utils.io import load_yaml

# Configure Streamlit page metadata and layout.
st.set_page_config(page_title="Lost in Pieces Demo", layout="wide")
# Render page title.
st.title("Lost in Pieces - Interactive Reconstruction Demo (D3)")

# Resolve default config path.
cfg_path = Path("configs/default.yaml")
# Guard when config is missing.
if not cfg_path.exists():
    st.error("Missing configs/default.yaml")
    st.stop()

# Load base configuration.
base_cfg = load_yaml(cfg_path)
# Resolve puzzle root directory.
puzzle_root = Path(base_cfg["data"]["puzzle_dir"])
# Discover generated puzzle folders.
puzzle_dirs = list_puzzle_dirs(puzzle_root)

# Guard when no generated puzzles are available.
if not puzzle_dirs:
    st.warning("No puzzles found. Run: image-analysis make-puzzles")
    st.stop()

# Render setup section header.
st.subheader("1) Setup")
# Create setup control columns.
col1, col2, col3 = st.columns(3)

with col1:
    # Select puzzle folder.
    puzzle_name = st.selectbox("Puzzle", [p.name for p in puzzle_dirs])
with col2:
    # Select experiment variant.
    variant = st.selectbox("Variant", ["combined", "classical_only", "deep_only"], index=0)
with col3:
    # Select execution mode.
    mode = st.selectbox("Run mode", ["Hybrid cached + live", "Live only", "Cached only"], index=0)

# Expose advanced solver controls.
with st.expander("Advanced Solver Controls", expanded=False):
    # Create solver-parameter columns.
    c1, c2 = st.columns(2)
    with c1:
        # Control number of local-search restarts.
        restarts = st.slider("Restarts", min_value=1, max_value=12, value=2)
    with c2:
        # Control iterations per restart.
        iterations = st.slider("Iterations", min_value=50, max_value=2000, value=250, step=50)

# Resolve selected puzzle path object.
selected_puzzle = next(p for p in puzzle_dirs if p.name == puzzle_name)
# Define dedicated D3 run root.
d3_root = Path("artifacts/runs/d3")
# Define legacy run root used by CLI experiments.
legacy_root = Path(base_cfg["evaluation"]["output_dir"])

# Render existing-run section header.
st.subheader("2) Load Existing")
# Search D3 runs first.
existing = find_existing_runs(d3_root, puzzle_name=puzzle_name, variant=variant)
# Fallback to legacy runs if no D3 runs found.
if not existing:
    existing = find_existing_runs(legacy_root, puzzle_name=puzzle_name, variant=variant)

# Show run selector when history exists.
if existing:
    selected_existing = st.selectbox("Existing run", [str(p) for p in existing], index=0)
else:
    # Keep null selected-existing marker when history is empty.
    selected_existing = None
    st.info("No existing run found for this puzzle/variant.")

# Render run section header.
st.subheader("3) Run")
# Render action button for live execution.
run_clicked = st.button("Run Reconstruction")

# Allow live execution only in modes that include live runs.
live_allowed = mode in ("Hybrid cached + live", "Live only")
# Allow cached loading only in modes that include cache reads.
load_cached = mode in ("Hybrid cached + live", "Cached only")

# Initialize result placeholders.
run_dir = None
run_meta = None
metrics = None
prediction = None
comparison_path = None
mismatch_path = None
stage_durations = None

# Execute live run when user clicked and mode allows it.
if run_clicked and live_allowed:
    # Render process section header.
    st.subheader("4) Process")
    # Reserve status placeholder.
    stage_box = st.empty()

    # Execute pipeline with spinner UX.
    with st.spinner("Running reconstruction pipeline..."):
        # Report config preparation stage.
        stage_box.info("Stage: load config and variant")
        # Build variant-adjusted config.
        vcfg = build_variant_config(base_cfg, variant)

        # Report expected pipeline stages before call.
        stage_box.info("Stage: extract features")
        stage_box.info("Stage: build compatibility")
        stage_box.info("Stage: solve + evaluate + save outputs")

        # Run full single-puzzle pipeline and persist timestamped artifacts.
        res = run_single_puzzle_variant(
            cfg=vcfg,
            puzzle_dir=selected_puzzle,
            variant=variant,
            output_root=d3_root,
            solver_overrides={"restarts": int(restarts), "iterations": int(iterations)},
            timestamped=True,
        )

    # Report completion state.
    stage_box.success("Completed all stages")
    # Save returned run folder path.
    run_dir = res.run_dir
    # Save metrics payload.
    metrics = res.metrics
    # Save prediction payload.
    prediction = res.prediction
    # Save comparison image path.
    comparison_path = res.comparison_path
    # Save mismatch overlay image path.
    mismatch_path = res.mismatch_path
    # Save stage-duration payload.
    stage_durations = res.stage_durations

# Otherwise load cached run when mode allows it.
elif load_cached and selected_existing is not None:
    # Resolve selected run directory path.
    run_dir = Path(selected_existing)

# Load persisted artifacts from run folder when needed.
if run_dir is not None and metrics is None:
    # Resolve metrics file path.
    mpath = run_dir / "metrics.json"
    # Resolve prediction file path.
    ppath = run_dir / "prediction.json"
    # Resolve metadata file path.
    meta_path = run_dir / "run_meta.json"

    # Load metrics JSON if present.
    if mpath.exists():
        metrics = json.loads(mpath.read_text(encoding="utf-8"))
    # Load prediction JSON if present.
    if ppath.exists():
        prediction = json.loads(ppath.read_text(encoding="utf-8"))
    # Load run metadata JSON if present.
    if meta_path.exists():
        run_meta = json.loads(meta_path.read_text(encoding="utf-8"))
        # Extract stage-duration map from metadata.
        stage_durations = run_meta.get("stage_durations")

    # Resolve comparison image path.
    comparison_path = run_dir / "comparison.png"
    # Resolve mismatch overlay path.
    mismatch_path = run_dir / "mismatch_overlay.png"

# Render results section when metrics are available.
if metrics is not None:
    # Render results section header.
    st.subheader("5) Results")
    # Create metric card columns.
    m1, m2, m3, m4 = st.columns(4)
    # Show piece-placement accuracy card.
    m1.metric("Piece Accuracy", f"{metrics.get('piece_accuracy', 0):.4f}")
    # Show neighbor accuracy card.
    m2.metric("Neighbor Accuracy", f"{metrics.get('neighbor_accuracy', 0):.4f}")
    # Show rotation accuracy card.
    m3.metric("Rotation Accuracy", f"{metrics.get('rotation_accuracy', 0):.4f}")
    # Show objective value card.
    m4.metric("Objective", f"{metrics.get('objective', 0):.4f}")

    # Render stage-duration diagnostics when available.
    if stage_durations:
        st.markdown("**Stage Durations (seconds)**")
        st.json(stage_durations)

    # Create side-by-side image columns.
    c1, c2 = st.columns(2)
    with c1:
        # Render comparison image if file exists.
        if comparison_path is not None and comparison_path.exists():
            st.image(str(comparison_path), caption="Ground Truth vs Reconstruction")
    with c2:
        # Render mismatch overlay if file exists.
        if mismatch_path is not None and mismatch_path.exists():
            st.image(str(mismatch_path), caption="Mismatch Overlay")

# Render artifact export section header.
st.subheader("6) Artifacts")
# Render export controls when a run folder is selected/produced.
if run_dir is not None:
    # Show run directory path.
    st.code(str(run_dir))
    # Resolve metrics artifact path.
    f1 = run_dir / "metrics.json"
    # Resolve prediction artifact path.
    f2 = run_dir / "prediction.json"
    # Resolve metadata artifact path.
    f3 = run_dir / "run_meta.json"
    # Create one download button per available JSON artifact.
    for f in (f1, f2, f3):
        if f.exists():
            st.download_button(
                label=f"Download {f.name}",
                data=f.read_bytes(),
                file_name=f.name,
                mime="application/json",
            )
else:
    # Prompt user when no run is loaded yet.
    st.info("Run or load an existing result to view/export artifacts.")

# Render reference links section.
st.markdown("### References")
# Link assignment deliverable checklist.
st.markdown("- Assignment deliverables: docs/IMAGE_ANALYSIS_ASSIGNMENT_DELIVERABLES.md")
# Link presentation deliverable checklist.
st.markdown("- Presentation deliverables: docs/IMAGE_ANALYSIS_PRESENTATION_DELIVERABLES.md")
