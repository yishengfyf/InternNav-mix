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
    patterns = [str(root / "**" / name), str(root / name)]
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
    progress=[json.loads(x) for x in (root/"progress.json").read_text().splitlines() if x.strip()] if (root/"progress.json").is_file() else []
    contexts=rows(root,"s2_recovery_context_events.jsonl")
    queries=rows(root,"replay_ledger/queries.jsonl")
    actions=rows(root,"replay_ledger/actions.jsonl")
    native=[r for r in contexts if r.get("event_type")=="stage65_native_recovery_set"]
    anchors=[]
    for r in native:
        report=r.get("stage59_productive_onset") or {}
        if report: anchors.append(report)
    recovery_queries=[r for r in queries if r.get("input_steps",{}).get("recovery_context") or r.get("stage65_native")]
    report={
      "task":"stage65_native_recovery_active",
      "expected_episode_count":a.expected_episodes,
      "completed_episode_count":len(progress),
      "native_recovery_set_count":len(native),
      "native_anchor_steps":[r.get("anchor_step") for r in native],
      "context_event_type_counts":dict(Counter(str(r.get("event_type")) for r in contexts)),
      "query_count":len(queries), "recovery_query_count":len(recovery_queries),
      "action_count":len(actions),
      "native_events":native,
      "natural_d0_event_count":len(anchors),
      "natural_d0_scene_count":len({str(r.get("scene_id")) for r in native}),
      "productive_anchor_count":sum(1 for r in anchors for x in (r.get("anchors") or []) if x.get("anchor")=="last_productive_pre_loop" and x.get("valid")),
      "integrity_passed":len(progress)==a.expected_episodes and len(native)==a.expected_episodes,
      "shadow_only":False,"decision_applied":True,"unknown_is_free":False,"gt_used_for_navigation":False,
    }
    a.output.parent.mkdir(parents=True,exist_ok=True); a.output.write_text(json.dumps(report,indent=2,ensure_ascii=False)+"\n"); print(json.dumps(report,indent=2,ensure_ascii=False))
if __name__=="__main__": main()
