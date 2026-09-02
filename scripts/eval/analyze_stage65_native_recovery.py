"""Audit Stage65 native recovery handoff and one-primitive execution."""
from __future__ import annotations
import argparse, glob, json
from collections import Counter
from pathlib import Path

def rows(root: Path, name: str):
    out=[]
    # Returns may preserve an extra ``run/`` component (and rank-specific
    # directories) while lightweight latest returns may flatten it.  Search
    # recursively so a missing top-level match cannot masquerade as zero S2
    # queries/actions.
    # Ledger files live below rank/episode directories, e.g.
    # ``replay_ledger/<scene_episode_rank>/queries.jsonl``.  The previous
    # ``**/replay_ledger/queries.jsonl`` pattern skipped that component and
    # silently reported zero queries/actions.  Match by basename while still
    # accepting flattened lightweight returns.
    patterns = [str(root / "**" / "replay_ledger" / "*" / Path(name).name),
                str(root / "**" / Path(name).name), str(root / name)]
    seen = set()
    for pattern in patterns:
      for path in glob.glob(pattern, recursive=True):
        if path in seen:
          continue
        seen.add(path)
        for line in Path(path).read_text(encoding="utf-8").splitlines():
            if line.strip(): out.append(json.loads(line))
    return out

def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--run-root",type=Path,required=True); ap.add_argument("--output",type=Path,required=True); ap.add_argument("--expected-episodes",type=int,required=True); a=ap.parse_args()
    root=a.run_root
    # Prefer the run-level progress ledger.  Visual bundles also contain one
    # progress file per rank (often only two records); selecting the first
    # rank file understated completion for multi-rank returns.
    progress_candidates = [root / "progress.json", root / "run" / "progress.json"]
    progress_path = next((p for p in progress_candidates if p.is_file()), None)
    if progress_path is None:
        candidates = sorted(root.glob("**/progress.json"))
        progress_path = candidates[0] if candidates else None
    progress=[json.loads(x) for x in progress_path.read_text().splitlines() if x.strip()] if progress_path else []
    contexts=rows(root,"s2_recovery_context_events.jsonl")
    queries=rows(root,"replay_ledger/queries.jsonl")
    actions=rows(root,"replay_ledger/actions.jsonl")
    native=[r for r in contexts if r.get("event_type")=="stage65_native_recovery_set"]
    native_episode_keys = {
        (str(r.get("scene_id")), str(r.get("episode_id"))) for r in native
    }
    anchors=[]
    for r in native:
        report=r.get("stage59_productive_onset") or {}
        if report: anchors.append(report)
    recovery_queries=[r for r in queries if (r.get("input_steps") or {}).get("recovery_context_active")
                      or (r.get("input_steps") or {}).get("stage65_native")
                      or r.get("stage65_native")]
    report={
      "task":"stage65_native_recovery_active",
      "expected_episode_count":a.expected_episodes,
      "completed_episode_count":len(progress),
      "native_recovery_set_count":len(native),
      "native_recovery_episode_count":len(native_episode_keys),
      "native_anchor_steps":[r.get("anchor_step") for r in native],
      "context_event_type_counts":dict(Counter(str(r.get("event_type")) for r in contexts)),
      "query_count":len(queries), "recovery_query_count":len(recovery_queries),
      "action_count":len(actions),
      "native_events":native,
      "natural_d0_event_count":len(anchors),
      "natural_d0_scene_count":len({str(r.get("scene_id")) for r in native}),
      "productive_anchor_count":sum(1 for r in anchors for x in (r.get("anchors") or []) if x.get("anchor")=="last_productive_pre_loop" and x.get("valid")),
      # A single episode may contain multiple natural D0 events.  Integrity is
      # episode-level, while native_recovery_set_count remains event-level.
      "integrity_passed":len(progress)==a.expected_episodes and len(native_episode_keys)==a.expected_episodes,
      "shadow_only":False,"decision_applied":True,"unknown_is_free":False,"gt_used_for_navigation":False,
    }
    a.output.parent.mkdir(parents=True,exist_ok=True); a.output.write_text(json.dumps(report,indent=2,ensure_ascii=False)+"\n"); print(json.dumps(report,indent=2,ensure_ascii=False))
if __name__=="__main__": main()
