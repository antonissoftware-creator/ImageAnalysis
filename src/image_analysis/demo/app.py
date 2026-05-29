from __future__ import annotations

# JSON parsing for summary metrics file.
import json
# Path handling for artifact discovery.
from pathlib import Path

# Streamlit UI framework.
import streamlit as st

# Render page title.
st.title("Lost in Pieces - Reconstruction Demo")

# Define default run output root.
run_root = Path("artifacts/runs/default")
# Define summary metric file path.
summary_file = run_root / "summary_metrics.json"

# Show experiment summary when available.
if summary_file.exists():
    # Parse summary JSON.
    summary = json.loads(summary_file.read_text(encoding="utf-8"))
    # Render summary section title.
    st.subheader("Experiment Summary")
    # Render raw summary JSON.
    st.json(summary)
else:
    # Show hint when no experiments have run yet.
    st.info("No summary yet. Run: image-analysis run-experiments")

# Scan and preview saved comparison figures.
if run_root.exists():
    # Discover all comparison images recursively.
    candidates = sorted([p for p in run_root.rglob("comparison.png")])
    # Show preview widget only if any figures were found.
    if candidates:
        # Render section title.
        st.subheader("Qualitative Results")
        # Let user pick one comparison image path.
        pick = st.selectbox("Comparison image", [str(p) for p in candidates])
        # Render selected image.
        st.image(pick, caption=pick)

# Render links to requirement checklists.
st.markdown("### References")
# Link assignment deliverable checklist.
st.markdown("- Assignment deliverables: docs/IMAGE_ANALYSIS_ASSIGNMENT_DELIVERABLES.md")
# Link presentation deliverable checklist.
st.markdown("- Presentation deliverables: docs/IMAGE_ANALYSIS_PRESENTATION_DELIVERABLES.md")
