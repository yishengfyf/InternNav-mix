import importlib.util
from pathlib import Path


_path = Path(__file__).resolve().parents[2] / "internnav" / "utils" / "stage41_executor_contract.py"
_spec = importlib.util.spec_from_file_location("stage41_executor_contract", _path)
_module = importlib.util.module_from_spec(_spec)
assert _spec.loader is not None
_spec.loader.exec_module(_module)


def _candidate(**extra):
    value = {"floor_aligned_known_free": True, "unknown_fraction": 0, "occupied_fraction": 0, "route_occ_conflict": False}
    value.update(extra)
    return value


def _edge(**extra):
    value = {"sparseocc_safe": True, "depth_occlusion_checked": True, "depth_readable": True, "unknown": False, "occupied": False}
    value.update(extra)
    return value


def test_contract_uses_sensor_hfov_and_readable_depth():
    report = _module.validate_executor_contract(
        sensor={"hfov_deg": 79, "hfov_source": "rgb_sensor", "depth_readable": True},
        edge_audits=[_edge(), _edge()],
        candidate_safety=_candidate(),
    )
    assert report["executor_eligible"] is True
    assert _module.executor_contract_ok(report)


def test_contract_abstains_without_depth_or_on_unknown_edge():
    report = _module.validate_executor_contract(
        sensor={"hfov_deg": 135, "hfov_source": "legacy_default", "depth_readable": False},
        edge_audits=[_edge(unknown=True)],
        candidate_safety=_candidate(),
    )
    assert report["executor_eligible"] is False
    assert report["action_emitted"] is False
    assert _module.executor_contract_ok(report)
