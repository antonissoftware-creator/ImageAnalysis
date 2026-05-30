from __future__ import annotations

# Path helpers for temporary filesystem test setup.
from pathlib import Path

# Runner utilities under test.
from image_analysis.core.runner import build_variant_config, find_existing_runs
# Shared JSON and directory helpers.
from image_analysis.utils.io import dump_json, ensure_dir


# Validate that deep-only variant disables classical families.
def test_build_variant_config_deep_only() -> None:
    # Build minimal config fixture.
    cfg = {
        "features": {
            "color": {"enabled": True},
            "texture": {"enabled": True},
            "deep": {"enabled": True},
        },
        "adjacency": {"family_weights": {"color": 1.0, "texture": 1.0, "deep": 1.0}},
    }
    # Build deep-only variant config.
    out = build_variant_config(cfg, "deep_only")
    # Assert color family is disabled.
    assert out["features"]["color"]["enabled"] is False
    # Assert texture family is disabled.
    assert out["features"]["texture"]["enabled"] is False
    # Assert color weight is zeroed.
    assert out["adjacency"]["family_weights"]["color"] == 0.0
    # Assert texture weight is zeroed.
    assert out["adjacency"]["family_weights"]["texture"] == 0.0


# Validate run discovery and filtering behavior.
def test_find_existing_runs_filters(tmp_path: Path) -> None:
    # Create temporary runs root.
    root = tmp_path / "runs"
    ensure_dir(root)

    # Create first fake run folder.
    r1 = root / "r1"
    # Create second fake run folder.
    r2 = root / "r2"
    ensure_dir(r1)
    ensure_dir(r2)

    # Write run metadata for first folder.
    dump_json(r1 / "run_meta.json", {"puzzle": "puzzle_000", "variant": "combined"})
    # Write run metadata for second folder.
    dump_json(r2 / "run_meta.json", {"puzzle": "puzzle_001", "variant": "deep_only"})

    # Query runs by first puzzle/variant.
    a = find_existing_runs(root, puzzle_name="puzzle_000", variant="combined")
    # Query runs by second puzzle/variant.
    b = find_existing_runs(root, puzzle_name="puzzle_001", variant="deep_only")

    # Assert one matching run was found for query A.
    assert len(a) == 1
    # Assert one matching run was found for query B.
    assert len(b) == 1
    # Assert query A returns expected folder.
    assert a[0].name == "r1"
    # Assert query B returns expected folder.
    assert b[0].name == "r2"
