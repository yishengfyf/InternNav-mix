import importlib.util
from pathlib import Path


_path = Path(__file__).resolve().parents[2] / "internnav" / "utils" / "stage58_support_policy.py"
_spec = importlib.util.spec_from_file_location("stage58_support_policy", _path)
_module = importlib.util.module_from_spec(_spec)
assert _spec.loader is not None
_spec.loader.exec_module(_module)


class _Memory:
    cs = 0.05

    def __init__(self):
        self.occ_counts = {}
        self.occ3d_frame_counts = {}
        self.free2d_counts = {
            (row, col): 2 for row in range(8, 13) for col in range(8, 18)
        }
        self.occ2d_counts = {}


def test_known_free_floor_fallback_is_read_only_and_policy_specific():
    memory = _Memory()
    report = _module.audit_support_policy_sweep(
        memory,
        [[10, 10], [10, 15]],
        runtime_contract={"agent_radius_m": 0.10},
        offline_primitive_truth={"valid": True, "primitive_safe": True},
    )
    arms = {arm["policy"]: arm for arm in report["arms"]}
    assert arms["observed_frames2"]["predicted_first_primitive_safe"] is False
    assert arms["known_free_floor_frames2"]["predicted_first_primitive_safe"] is True
    assert report["decision_applied"] is False
    assert report["unknown_is_free"] is False
    assert report["known_free_floor_fallback_mutates_memory"] is False
    assert memory.occ_counts == {}


def test_occupied_2d_cell_is_not_floor_fallback_support():
    memory = _Memory()
    memory.occ2d_counts[(10, 13)] = 1
    report = _module.audit_support_policy_sweep(
        memory,
        [[10, 10], [10, 15]],
        runtime_contract={},
        offline_primitive_truth={"valid": True, "primitive_safe": True},
    )
    arm = next(
        item for item in report["arms"] if item["policy"] == "known_free_floor_frames2"
    )
    assert arm["predicted_first_primitive_safe"] is False
