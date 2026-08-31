import importlib.util
from pathlib import Path


_path = Path(__file__).resolve().parents[2] / "internnav" / "utils" / "stage58_geometry_contract.py"
_spec = importlib.util.spec_from_file_location("stage58_geometry_contract", _path)
_module = importlib.util.module_from_spec(_spec)
assert _spec.loader is not None
_spec.loader.exec_module(_module)


class _Memory:
    cs = 0.05

    def __init__(self):
        self.occ_counts = {}
        self.occ3d_frame_counts = {}
        for col in range(5, 21):
            for row in range(5, 16):
                key = (row, col, 0)
                self.occ_counts[key] = 3
                self.occ3d_frame_counts[key] = 2


def test_radius_sweep_is_shadow_only_and_tracks_truth_errors():
    report = _module.audit_geometry_radius_sweep(
        _Memory(),
        [[10, 10], [10, 15]],
        footprint_radii_m=[0.18, 0.10, 0.15, 0.13],
        runtime_contract={"agent_radius_m": 0.10},
        offline_primitive_truth={"valid": True, "primitive_safe": True},
    )
    assert report["decision_applied"] is False
    assert report["pixel_translation_allowed"] is False
    assert report["unknown_is_free"] is False
    assert [arm["footprint_radius_m"] for arm in report["arms"]] == [0.10, 0.13, 0.15, 0.18]
    assert all(arm["predicted_first_primitive_safe"] for arm in report["arms"])
    assert all(not arm["false_safe"] for arm in report["arms"])


def test_radius_sweep_marks_unsafe_truth_as_false_safe():
    report = _module.audit_geometry_radius_sweep(
        _Memory(),
        [[10, 10], [10, 15]],
        footprint_radii_m=[0.10],
        runtime_contract={},
        offline_primitive_truth={"valid": True, "primitive_safe": False},
    )
    assert report["arms"][0]["false_safe"] is True
