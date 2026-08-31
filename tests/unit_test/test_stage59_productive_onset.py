from internnav.utils.stage59_productive_onset import audit_productive_onset_anchors


def _raster(start, end, **_kwargs):
    row0, col0 = start
    row1, col1 = end
    steps = max(abs(row1 - row0), abs(col1 - col0), 1)
    return [
        (
            round(row0 + (row1 - row0) * index / steps),
            round(col0 + (col1 - col0) * index / steps),
        )
        for index in range(steps + 1)
    ]


def test_selects_causal_onset_and_productive_pre_loop_anchor():
    trace = [
        {"step_id": 0, "row": 0, "col": 0, "x": 0.0, "y": 0.0},
        {"step_id": 1, "row": 0, "col": 5, "x": 0.0, "y": 0.25},
        {"step_id": 2, "row": 0, "col": 10, "x": 0.0, "y": 0.50},
        {"step_id": 8, "row": 0, "col": 10, "x": 0.0, "y": 0.50},
    ]
    result = audit_productive_onset_anchors(
        trace,
        trigger_step=8,
        onset_step=2,
        state_fn=lambda _row, _col: "free",
        rasterize_edge=_raster,
        cell_size_m=0.05,
    )
    anchors = {row["anchor"]: row for row in result["anchors"]}
    assert anchors["raw_trigger"]["step_id"] == 8
    assert anchors["estimated_loop_onset"]["step_id"] == 2
    assert anchors["last_productive_pre_loop"]["step_id"] == 1
    assert anchors["last_productive_pre_loop"]["route_edge_count"] == 1
    assert anchors["last_productive_pre_loop"]["first_edge"]["safe_0p25m_prefix"]
    assert result["decision_applied"] is False
    assert result["unknown_is_free"] is False


def test_unknown_prefix_is_not_safe_and_no_history_is_explicit():
    trace = [{"step_id": 7, "row": 2, "col": 3, "x": 0.0, "y": 0.0}]
    result = audit_productive_onset_anchors(
        trace,
        trigger_step=7,
        onset_step=7,
        state_fn=lambda _row, _col: "unknown",
        rasterize_edge=_raster,
    )
    anchors = {row["anchor"]: row for row in result["anchors"]}
    assert anchors["raw_trigger"]["route_edge_count"] == 0
    assert not anchors["last_productive_pre_loop"]["valid"]
    assert anchors["last_productive_pre_loop"]["reason"] == "no_pre_loop_translation_edge"
