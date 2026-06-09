"""
Post-hoc directional feature analysis for Stage14a-v2 design.

Reads Stage14a results and computes direction-aware features from
trajectory_events.jsonl (pixel_goal eccentricity, GPS heading consistency,
compass reversal count) then runs AUC + FPR-constrained threshold search
to assess whether direction signals can separate failure from success.

Also runs a "semantic backtrack value" post-hoc:
  For each failure episode, find the last semantically advancing keyframe
  position and estimate whether backtracking there would reduce NE.

Usage:
    python analyze_direction_features.py \
        --run-dir ./compare_stage14a_failure_prediction_shadow_epseed_100 \
        --output direction_analysis.json
"""

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------

def _read_jsonl(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        return []
    records = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            records.append(json.loads(line))
    return records


def _safe_float(v: Any, default: float = 0.0) -> float:
    try:
        return float(v) if v is not None else default
    except (TypeError, ValueError):
        return default


def _safe_int(v: Any, default: int = 0) -> int:
    try:
        return int(v) if v is not None else default
    except (TypeError, ValueError):
        return default


def _episode_key(r: Dict[str, Any]) -> str:
    return f"{r.get('scene_id')}|{r.get('episode_id')}"


def _resolve_run_dir(path: Path) -> Path:
    for candidate in [path, path / "vlmap_safety_debug" / "run_001"]:
        if (candidate / "trajectory_events.jsonl").exists():
            return candidate
    raise FileNotFoundError(f"trajectory_events.jsonl not found under {path}")


def _group_by_episode(records: List[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
    grouped: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for r in records:
        grouped[_episode_key(r)].append(r)
    for items in grouped.values():
        items.sort(key=lambda x: _safe_int(x.get("eval_step", x.get("step_id")), -1))
    return grouped


# ---------------------------------------------------------------------------
# Direction feature extraction
# ---------------------------------------------------------------------------

IMAGE_WIDTH = 640
IMAGE_CENTER_X = IMAGE_WIDTH / 2.0  # 320


def _pixel_eccentricity(pixel_goal: Any) -> Optional[float]:
    """|x - center| / half_width -> [0, 1]"""
    if not pixel_goal or len(pixel_goal) < 1:
        return None
    try:
        x = float(pixel_goal[0])
    except (TypeError, ValueError):
        return None
    return abs(x - IMAGE_CENTER_X) / IMAGE_CENTER_X


def _pixel_side(pixel_goal: Any) -> Optional[str]:
    """'left' / 'right' / 'center'"""
    if not pixel_goal or len(pixel_goal) < 1:
        return None
    try:
        x = float(pixel_goal[0])
    except (TypeError, ValueError):
        return None
    if x < IMAGE_CENTER_X - 32:
        return "left"
    if x > IMAGE_CENTER_X + 32:
        return "right"
    return "center"


def _heading_deg(gps_prev: List[float], gps_curr: List[float]) -> Optional[float]:
    """Heading in degrees from gps_prev to gps_curr (x, z plane)."""
    dx = gps_curr[0] - gps_prev[0]
    dz = gps_curr[1] - gps_prev[1]
    dist = math.sqrt(dx * dx + dz * dz)
    if dist < 1e-4:
        return None
    return math.degrees(math.atan2(dx, dz))


def _circular_variance(angles_deg: List[float]) -> float:
    """Circular variance in [0, 1] (0 = all same direction, 1 = maximally spread)."""
    if len(angles_deg) < 2:
        return 0.0
    rads = [math.radians(a) for a in angles_deg]
    sin_mean = sum(math.sin(r) for r in rads) / len(rads)
    cos_mean = sum(math.cos(r) for r in rads) / len(rads)
    r_bar = math.sqrt(sin_mean ** 2 + cos_mean ** 2)
    return 1.0 - r_bar  # 0 = consistent, 1 = random


def _max_consecutive_same_side(sides: List[str]) -> int:
    """Longest run of consecutive 'left' or 'right' (ignoring 'center')."""
    directional = [s for s in sides if s in ("left", "right")]
    if not directional:
        return 0
    max_run = 1
    cur_run = 1
    for i in range(1, len(directional)):
        if directional[i] == directional[i - 1]:
            cur_run += 1
            max_run = max(max_run, cur_run)
        else:
            cur_run = 1
    return max_run


def _compass_reversals(compass_vals: List[float]) -> int:
    """Number of sign changes in compass (bearing) within a window."""
    if len(compass_vals) < 2:
        return 0
    reversals = 0
    for i in range(1, len(compass_vals)):
        if compass_vals[i - 1] * compass_vals[i] < 0:
            reversals += 1
    return reversals


def extract_direction_features(
    traj_events: List[Dict[str, Any]],
    step: int,
    window: int,
) -> Dict[str, float]:
    """Compute direction features from trajectory_events up to `step`."""
    # Filter events up to the target step
    past = [e for e in traj_events if _safe_int(e.get("eval_step", -1)) <= step]
    win_start = max(0, step - window)
    window_events = [e for e in past if _safe_int(e.get("eval_step", -1)) >= win_start]

    if not past:
        return _zero_direction_features()

    # --- pixel_goal features ---
    pg_eccentricities = []
    pg_sides = []
    for e in window_events:
        pg = e.get("pixel_goal")
        ecc = _pixel_eccentricity(pg)
        side = _pixel_side(pg)
        if ecc is not None:
            pg_eccentricities.append(ecc)
        if side is not None:
            pg_sides.append(side)

    pg_ecc_mean = sum(pg_eccentricities) / len(pg_eccentricities) if pg_eccentricities else 0.0
    pg_ecc_max = max(pg_eccentricities) if pg_eccentricities else 0.0
    pg_side_streak = _max_consecutive_same_side(pg_sides) / max(1, len(pg_sides))
    pg_side_streak_raw = _max_consecutive_same_side(pg_sides)

    # --- GPS heading features ---
    gps_list = []
    for e in window_events:
        gps = e.get("gps")
        if gps and len(gps) >= 2:
            gps_list.append((
                _safe_int(e.get("eval_step", -1)),
                [_safe_float(gps[0]), _safe_float(gps[1])],
            ))
    gps_list.sort(key=lambda x: x[0])

    headings = []
    for i in range(1, len(gps_list)):
        h = _heading_deg(gps_list[i - 1][1], gps_list[i][1])
        if h is not None:
            headings.append(h)

    heading_variance = _circular_variance(headings) if headings else 0.0
    heading_consistency = 1.0 - heading_variance

    # Total displacement over the window (for context)
    total_displacement = 0.0
    for i in range(1, len(gps_list)):
        dx = gps_list[i][1][0] - gps_list[i - 1][1][0]
        dz = gps_list[i][1][1] - gps_list[i - 1][1][1]
        total_displacement += math.sqrt(dx * dx + dz * dz)

    # --- Compass features ---
    compass_vals = []
    for e in window_events:
        c = e.get("compass")
        if c and len(c) > 0:
            compass_vals.append(_safe_float(c[0]))

    compass_reversal_count = _compass_reversals(compass_vals)
    compass_reversal_rate = compass_reversal_count / max(1, len(compass_vals))

    # Compass range (spread)
    compass_range = 0.0
    if compass_vals:
        compass_range = max(compass_vals) - min(compass_vals)

    # --- Combined: eccentricity * (1 - heading_consistency) ---
    directionality_confusion = pg_ecc_mean * (1.0 - heading_consistency + 1e-3)

    return {
        "pg_ecc_mean_w": pg_ecc_mean,
        "pg_ecc_max_w": pg_ecc_max,
        "pg_side_streak_ratio_w": pg_side_streak,
        "pg_side_streak_raw_w": float(pg_side_streak_raw),
        "heading_variance_w": heading_variance,
        "heading_consistency_w": heading_consistency,
        "compass_reversal_count_w": float(compass_reversal_count),
        "compass_reversal_rate_w": compass_reversal_rate,
        "compass_range_w": compass_range,
        "directionality_confusion_w": directionality_confusion,
        "traj_displacement_w": total_displacement,
    }


def _zero_direction_features() -> Dict[str, float]:
    return {
        "pg_ecc_mean_w": 0.0,
        "pg_ecc_max_w": 0.0,
        "pg_side_streak_ratio_w": 0.0,
        "pg_side_streak_raw_w": 0.0,
        "heading_variance_w": 0.0,
        "heading_consistency_w": 1.0,
        "compass_reversal_count_w": 0.0,
        "compass_reversal_rate_w": 0.0,
        "compass_range_w": 0.0,
        "directionality_confusion_w": 0.0,
        "traj_displacement_w": 0.0,
    }


# ---------------------------------------------------------------------------
# OccMem features from failure_prediction_events (v14a)
# ---------------------------------------------------------------------------

def extract_fp_features(
    fp_events: List[Dict[str, Any]],
    step: int,
    window: int,
) -> Dict[str, float]:
    """Pull per-step features from failure_prediction_events.jsonl."""
    past = [e for e in fp_events if _safe_int(e.get("step_id", -1)) <= step]
    if not past:
        return {}
    current = past[-1]
    win_start = max(0, step - window)
    window_events = [e for e in past if _safe_int(e.get("step_id", -1)) >= win_start]
    if not window_events:
        window_events = [current]

    current_features = current.get("features", {})
    sb = current.get("signal_breakdown", {})

    # Running max of failure_score in window
    scores = [_safe_float(e.get("failure_score")) for e in window_events]
    max_score_w = max(scores) if scores else 0.0
    mean_score_w = sum(scores) / len(scores) if scores else 0.0

    return {
        "fp_stagnation_score": _safe_float(sb.get("stagnation_score")),
        "fp_semantic_score": _safe_float(sb.get("semantic_score")),
        "fp_collision_score": _safe_float(sb.get("collision_score")),
        "fp_displacement_score": _safe_float(sb.get("displacement_score")),
        "fp_failure_score_current": _safe_float(current.get("failure_score")),
        "fp_failure_score_max_w": max_score_w,
        "fp_failure_score_mean_w": mean_score_w,
        "fp_stagnation_streak": _safe_float(current_features.get("window_span_steps")),
        "fp_occ_growth_w": _safe_float(current_features.get("occ_growth_last_w")),
        "fp_displacement_w": _safe_float(current_features.get("displacement_total_w")),
        "fp_collision_sum_w": _safe_float(current_features.get("collision_sum_w")),
    }


# ---------------------------------------------------------------------------
# AUC & threshold search
# ---------------------------------------------------------------------------

def _auc(labels: List[int], values: List[float]) -> Optional[float]:
    pos = [v for l, v in zip(labels, values) if l == 1]
    neg = [v for l, v in zip(labels, values) if l == 0]
    if not pos or not neg:
        return None
    wins = sum(
        1.0 if p > n else 0.5 if p == n else 0.0
        for p in pos
        for n in neg
    )
    return wins / (len(pos) * len(neg))


def _threshold_search(
    labels: List[int],
    values: List[float],
    direction: str,
    max_fpr: float,
) -> Dict[str, Any]:
    n_fail = sum(labels)
    n_succ = len(labels) - n_fail
    best: Optional[Dict[str, Any]] = None
    for thresh in sorted(set(values)):
        preds = [v >= thresh if direction == "high" else v <= thresh for v in values]
        tp = sum(1 for p, l in zip(preds, labels) if p and l == 1)
        fp = sum(1 for p, l in zip(preds, labels) if p and l == 0)
        predicted = sum(preds)
        recall = tp / max(1, n_fail)
        fpr = fp / max(1, n_succ)
        precision = tp / max(1, predicted)
        if fpr > max_fpr:
            continue
        item = {
            "threshold": thresh, "direction": direction,
            "tp": tp, "fp": fp, "predicted": predicted,
            "recall": recall, "fpr": fpr, "precision": precision,
        }
        if best is None or (recall, precision) > (best["recall"], best["precision"]):
            best = item
    return best or {"threshold": None, "direction": direction, "tp": 0, "fp": 0,
                    "predicted": 0, "recall": 0.0, "fpr": 0.0, "precision": 0.0}


def _analyze_feature(
    name: str,
    rows: List[Dict[str, Any]],
    max_fpr: float,
) -> Dict[str, Any]:
    labels = [r["label"] for r in rows]
    values = [_safe_float(r["features"].get(name)) for r in rows]
    auc_hi = _auc(labels, values)
    auc_lo = None if auc_hi is None else 1.0 - auc_hi
    best_dir = "high" if (auc_hi or 0) >= (auc_lo or 0) else "low"
    best_auc = max(auc_hi or 0, auc_lo or 0) or None
    best_rule = _threshold_search(labels, values, best_dir, max_fpr)

    # Exclude ep160-equivalent (mode A) to measure B/C/D-only AUC
    non_a_rows = [r for r in rows if r.get("mode") != "stuck_wall_hugging"]
    non_a_labels = [r["label"] for r in non_a_rows]
    non_a_values = [_safe_float(r["features"].get(name)) for r in non_a_rows]
    auc_non_a_hi = _auc(non_a_labels, non_a_values)
    auc_non_a = None if auc_non_a_hi is None else max(auc_non_a_hi, 1.0 - auc_non_a_hi)
    best_rule_non_a = None
    if non_a_rows:
        best_rule_non_a = _threshold_search(non_a_labels, non_a_values, best_dir, max_fpr)

    # Mean by mode
    mode_means: Dict[str, float] = {}
    modes_seen = {r.get("mode") for r in rows}
    for mode in sorted(m for m in modes_seen if m):
        mode_vals = [_safe_float(r["features"].get(name)) for r in rows if r.get("mode") == mode]
        mode_means[mode] = sum(mode_vals) / len(mode_vals) if mode_vals else 0.0

    return {
        "feature": name,
        "auc": best_auc,
        "auc_direction": best_dir,
        "auc_without_mode_a": auc_non_a,
        "best_rule": best_rule,
        "best_rule_without_mode_a": best_rule_non_a,
        "mode_means": mode_means,
    }


# ---------------------------------------------------------------------------
# Semantic backtrack post-hoc
# ---------------------------------------------------------------------------

INSTRUCTION_ORDER = {
    "hallway": 0, "corridor": 0, "entrance": 0,
    "kitchen": 1, "dining room": 2, "dining": 2,
    "living room": 3, "fireplace": 3, "couch": 3, "sofa": 3,
    "bedroom": 4, "bathroom": 4, "office": 4,
    "door": 0,
}


def _semantic_progress(top_match: Optional[str]) -> int:
    if not top_match:
        return -1
    for key, rank in INSTRUCTION_ORDER.items():
        if key in top_match.lower():
            return rank
    return -1


def analyze_semantic_backtrack(
    mem_events: List[Dict[str, Any]],
    traj_events: List[Dict[str, Any]],
    progress_records: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """
    For each failure episode, find the last position where semantic progress
    was highest (most advanced landmark matched), and compute:
    - Distance back to that anchor from where stagnation occurred
    - Whether the anchor is plausibly closer to goal than the final position
    """
    sem_by_ep = defaultdict(list)
    for e in mem_events:
        if e.get("event_type") == "occ_memory_semantic":
            sem_by_ep[_episode_key(e)].append(e)

    traj_by_ep = defaultdict(list)
    for e in traj_events:
        traj_by_ep[_episode_key(e)].append(e)

    results = []
    for prog in progress_records:
        if _safe_float(prog.get("success")) >= 0.5:
            continue
        key = _episode_key(prog)
        sem_evs = sorted(sem_by_ep.get(key, []), key=lambda x: _safe_int(x.get("step_id"), -1))
        traj_evs = sorted(traj_by_ep.get(key, []), key=lambda x: _safe_int(x.get("eval_step"), -1))

        if not sem_evs or not traj_evs:
            continue

        # Find semantically "most advanced" position
        best_progress = -1
        best_anchor = None
        for ev in sem_evs:
            rank = _semantic_progress(ev.get("top_match"))
            if rank > best_progress:
                best_progress = rank
                best_anchor = ev

        # Find stagnation trigger (first semantic stagnation event)
        stagnation_step = None
        for ev in sem_evs:
            if ev.get("stagnation_would_requery"):
                stagnation_step = _safe_int(ev.get("step_id"))
                break

        # Final GPS position
        final_gps = traj_evs[-1].get("gps", []) if traj_evs else []

        # Anchor GPS (nearest traj event to anchor step)
        anchor_gps = None
        if best_anchor:
            anchor_xy = best_anchor.get("pose_xy", [])
            if anchor_xy and len(anchor_xy) >= 2:
                anchor_gps = anchor_xy  # already in world coords

        # Stagnation GPS
        stag_gps = None
        if stagnation_step is not None:
            candidates = [e for e in traj_evs if _safe_int(e.get("eval_step", -1)) <= stagnation_step]
            if candidates:
                stag_ev = candidates[-1]
                stag_gps_raw = stag_ev.get("gps", [])
                if stag_gps_raw and len(stag_gps_raw) >= 2:
                    stag_gps = [_safe_float(stag_gps_raw[0]), _safe_float(stag_gps_raw[1])]

        # Backtrack distance (anchor -> stagnation position)
        backtrack_dist = None
        if anchor_gps and stag_gps:
            dx = stag_gps[0] - _safe_float(anchor_gps[0])
            dz = stag_gps[1] - _safe_float(anchor_gps[1])
            backtrack_dist = math.sqrt(dx * dx + dz * dz)

        results.append({
            "episode_id": prog.get("episode_id"),
            "steps": _safe_int(prog.get("steps")),
            "final_ne": _safe_float(prog.get("ne")),
            "best_semantic_match": best_anchor.get("top_match") if best_anchor else None,
            "best_semantic_score": _safe_float(best_anchor.get("top_score")) if best_anchor else 0.0,
            "best_semantic_step": _safe_int(best_anchor.get("step_id")) if best_anchor else None,
            "best_anchor_pose": anchor_gps,
            "stagnation_step": stagnation_step,
            "stagnation_gps": stag_gps,
            "backtrack_distance_m": backtrack_dist,
            "final_gps": [_safe_float(g) for g in final_gps] if final_gps else None,
        })
    return results


# ---------------------------------------------------------------------------
# Main analysis
# ---------------------------------------------------------------------------

def _failure_mode(prog: Dict[str, Any]) -> str:
    if _safe_float(prog.get("success")) >= 0.5:
        return "success"
    steps = _safe_int(prog.get("steps"))
    ne = _safe_float(prog.get("ne"))
    collision = _safe_float(prog.get("collision_count"))
    stag = _safe_int(prog.get("occ_memory_recovery_max_occupied_stagnation_streak"))
    if stag >= 200 or (steps >= 300 and collision > 100):
        return "stuck_wall_hugging"
    if steps < 60:
        return "early_lost"
    if ne > 8.0:
        return "navigation_lost"
    return "mid_distance_fail"


def analyze(
    run_dir: Path,
    timepoints: List[int],
    window: int,
    max_fpr: float,
    max_steps: int,
) -> Dict[str, Any]:
    run_dir = _resolve_run_dir(run_dir)
    progress = _read_jsonl(run_dir / "progress.json")
    traj_events = _read_jsonl(run_dir / "trajectory_events.jsonl")
    fp_events = _read_jsonl(run_dir / "failure_prediction_events.jsonl")
    mem_events = _read_jsonl(run_dir / "occ_memory" / "memory_events.jsonl")

    prog_by_key = {_episode_key(p): p for p in progress}
    traj_by_key = _group_by_episode(traj_events)
    fp_by_key = _group_by_episode(fp_events)

    result: Dict[str, Any] = {
        "run_dir": str(run_dir),
        "episode_count": len(progress),
        "failure_count": sum(1 for p in progress if _safe_float(p.get("success")) < 0.5),
        "success_count": sum(1 for p in progress if _safe_float(p.get("success")) >= 0.5),
        "window": window,
        "max_fpr": max_fpr,
        "timepoints": {},
    }

    for step in timepoints:
        rows = []
        for key, prog in prog_by_key.items():
            if _safe_int(prog.get("steps")) < step:
                continue  # episode ended before this timepoint

            dir_feats = extract_direction_features(
                traj_by_key.get(key, []), step, window
            )
            fp_feats = extract_fp_features(
                fp_by_key.get(key, []), step, window
            )
            all_feats = {**dir_feats, **fp_feats}
            if not all_feats:
                continue

            rows.append({
                "key": key,
                "episode_id": prog.get("episode_id"),
                "label": 0 if _safe_float(prog.get("success")) >= 0.5 else 1,
                "mode": _failure_mode(prog),
                "features": all_feats,
            })

        if not rows:
            continue

        labels = [r["label"] for r in rows]
        feature_names = sorted(rows[0]["features"].keys())
        feature_results = [_analyze_feature(n, rows, max_fpr) for n in feature_names]
        feature_results.sort(key=lambda x: -(x["auc"] or 0))

        # Summary stats by mode
        mode_dist: Dict[str, int] = defaultdict(int)
        for r in rows:
            mode_dist[r["mode"]] += 1

        result["timepoints"][str(step)] = {
            "sample_count": len(rows),
            "failure_count": sum(labels),
            "success_count": len(labels) - sum(labels),
            "mode_distribution": dict(mode_dist),
            "feature_analysis": feature_results,
        }

    # Semantic backtrack post-hoc
    result["semantic_backtrack"] = analyze_semantic_backtrack(
        mem_events, traj_events, progress
    )

    return result


# ---------------------------------------------------------------------------
# Pretty printer
# ---------------------------------------------------------------------------

def _print_summary(result: Dict[str, Any]) -> None:
    print(f"\n{'='*70}")
    print(f"Direction Feature Analysis — {result['run_dir']}")
    print(f"Episodes: {result['episode_count']}  "
          f"Failure: {result['failure_count']}  "
          f"Success: {result['success_count']}  "
          f"Window: {result['window']}  max_FPR: {result['max_fpr']}")
    print(f"{'='*70}\n")

    for step_str, tp in sorted(result["timepoints"].items(), key=lambda x: int(x[0])):
        step = int(step_str)
        print(f"--- Timepoint step={step}  "
              f"(n={tp['sample_count']}, fail={tp['failure_count']}, "
              f"succ={tp['success_count']}) ---")
        print(f"Mode dist: {tp['mode_distribution']}")
        print(f"{'Feature':<38} {'AUC':>6} {'AUCnoA':>7} {'Recall':>7} {'FPR':>6} {'Thresh':>8}  Direction")
        print("-" * 90)
        for fa in tp["feature_analysis"][:20]:
            auc = fa.get("auc")
            auc_noa = fa.get("auc_without_mode_a")
            br = fa.get("best_rule") or {}
            recall = br.get("recall", 0)
            fpr = br.get("fpr", 0)
            thresh = br.get("threshold")
            direction = fa.get("auc_direction", "?")
            auc_str = f"{auc:.3f}" if auc is not None else "  N/A"
            auc_noa_str = f"{auc_noa:.3f}" if auc_noa is not None else "   N/A"
            thresh_str = f"{thresh:.3f}" if thresh is not None else "    N/A"
            print(f"  {fa['feature']:<36} {auc_str:>6} {auc_noa_str:>7} "
                  f"{recall:>7.3f} {fpr:>6.3f} {thresh_str:>8}  {direction}")
        print()

    # Semantic backtrack summary
    bt = result.get("semantic_backtrack", [])
    if bt:
        print(f"\n--- Semantic Backtrack Post-hoc ({len(bt)} failure episodes) ---")
        print(f"{'EpID':>6} {'Steps':>5} {'NE':>6} {'BestMatch':>15} "
              f"{'AnchorStep':>10} {'BacktrackDist':>13} {'StagnStep':>10}")
        print("-" * 80)
        for b in sorted(bt, key=lambda x: x.get("final_ne", 0), reverse=True):
            ep = b.get("episode_id", "?")
            steps = b.get("steps", "?")
            ne = b.get("final_ne", 0)
            match = b.get("best_semantic_match") or "?"
            anchor_step = b.get("best_semantic_step", "?")
            bt_dist = b.get("backtrack_distance_m")
            stag_step = b.get("stagnation_step", "?")
            bt_str = f"{bt_dist:.2f}m" if bt_dist is not None else "    N/A"
            print(f"  {ep:>4}  {steps:>5}  {ne:>6.2f}  {match:>15}  "
                  f"{str(anchor_step):>10}  {bt_str:>13}  {str(stag_step):>10}")

        # Summary: how many failures have a backtrack anchor within 3m?
        with_bt = [b for b in bt if b.get("backtrack_distance_m") is not None]
        short_bt = [b for b in with_bt if b["backtrack_distance_m"] <= 3.0]
        print(f"\nHave computable backtrack distance: {len(with_bt)}/{len(bt)}")
        print(f"Backtrack distance <= 3m: {len(short_bt)}/{len(with_bt)}")
        if with_bt:
            dists = [b["backtrack_distance_m"] for b in with_bt]
            print(f"Backtrack dist stats: mean={sum(dists)/len(dists):.2f}m, "
                  f"min={min(dists):.2f}m, max={max(dists):.2f}m")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Post-hoc directional feature analysis for Stage14a-v2 design."
    )
    parser.add_argument("--run-dir", type=Path, required=True,
                        help="Path to Stage14a result directory (or vlmap_safety_debug/run_001).")
    parser.add_argument("--timepoints", default="20,30,40,50,60",
                        help="Comma-separated step timepoints to analyze (default: 20,30,40,50,60).")
    parser.add_argument("--window", type=int, default=20,
                        help="Feature window in steps (default: 20).")
    parser.add_argument("--max-fpr", type=float, default=0.05,
                        help="Max allowed success FPR when searching thresholds (default: 0.05).")
    parser.add_argument("--max-steps", type=int, default=500)
    parser.add_argument("--output", type=Path,
                        help="Write full JSON results to this file.")
    parser.add_argument("--quiet", action="store_true",
                        help="Suppress pretty-printed summary.")
    args = parser.parse_args()

    timepoints = [int(t.strip()) for t in args.timepoints.split(",") if t.strip()]
    result = analyze(
        args.run_dir,
        timepoints=timepoints,
        window=args.window,
        max_fpr=args.max_fpr,
        max_steps=args.max_steps,
    )
    if not args.quiet:
        _print_summary(result)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n",
                               encoding="utf-8")
        print(f"\nFull results saved to {args.output}")


if __name__ == "__main__":
    main()
