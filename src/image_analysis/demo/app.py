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


# Inject custom CSS for right-aligned instant tooltips.
st.markdown(
    """
<style>
.step-header-row {
  display: flex;
  align-items: center;
  justify-content: space-between;
  width: 100%;
  margin: 0.25rem 0 0.5rem 0;
}
.step-header-title {
  font-size: 1.3rem;
  font-weight: 600;
  line-height: 1.35;
  margin: 0;
}
.step-tooltip-wrap {
  position: relative;
  display: inline-flex;
  align-items: center;
  justify-content: center;
}
.step-tooltip-icon {
  width: 16px;
  height: 16px;
  border-radius: 50%;
  font-size: 11px;
  font-weight: 700;
  line-height: 16px;
  text-align: center;
  background: #eef2ff;
  color: #1d4ed8;
  border: 1px solid #bfdbfe;
  cursor: help;
  user-select: none;
}
.step-tooltip-text {
  visibility: hidden;
  opacity: 0;
  transition: opacity 0s linear;
  position: absolute;
  right: 0;
  top: 22px;
  z-index: 9999;
  width: 360px;
  max-width: min(360px, 90vw);
  background: #111827;
  color: #f9fafb;
  text-align: left;
  border-radius: 6px;
  padding: 8px 10px;
  font-size: 12px;
  line-height: 1.35;
  box-shadow: 0 4px 12px rgba(0,0,0,0.2);
}
.step-tooltip-wrap:hover .step-tooltip-text {
  visibility: visible;
  opacity: 1;
}
</style>
""",
    unsafe_allow_html=True,
)


# Render a step header with a tooltip-style info icon.
def step_header(title: str, help_text: str) -> None:
    # Draw a full-width row with title on left and icon at far right.
    st.markdown(
        f"<div class='step-header-row'>"
        f"  <div class='step-header-title'>{title}</div>"
        f"  <div class='step-tooltip-wrap'>"
        f"    <span class='step-tooltip-icon'>!</span>"
        f"    <div class='step-tooltip-text'>{help_text}</div>"
        f"  </div>"
        f"</div>",
        unsafe_allow_html=True,
    )

# Configure Streamlit page metadata and layout.
st.set_page_config(page_title="Lost in Pieces Demo", layout="wide")
# Render page title.
st.title("Lost in Pieces - Interactive Reconstruction Demo")

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

# Render setup section header with tooltip.
step_header(
    "1) Setup",
    "Choose puzzle, variant, and solver settings. Used before Run Reconstruction starts. Not a compute stage by itself.",
)
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

# Render existing-run section header with tooltip.
step_header(
    "2) Load Existing",
    "Loads prior artifacts from disk when available. Used directly in Cached mode, and as fallback in Hybrid mode.",
)
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

# Render run section header with tooltip.
step_header(
    "3) Run",
    "Triggers full pipeline execution when clicked and mode allows live runs. In Cached-only mode, no live compute is triggered.",
)
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
    # Render process section header with tooltip.
    step_header(
        "4) Process",
        "Live execution stages run here: extract features, build compatibility, solve, evaluate, and save outputs.",
    )
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
    # Render results section header with tooltip.
    step_header(
        "5) Results",
        "Displays metrics and qualitative images from either a live run or loaded cached artifacts.",
    )
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

# Render artifact export section header with tooltip.
step_header(
    "6) Artifacts",
    "Shows run folder path and lets you download JSON artifacts. Populated after live run or cached run load.",
)
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
