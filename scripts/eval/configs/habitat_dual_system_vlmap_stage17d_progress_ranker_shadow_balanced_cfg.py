import copy
import importlib.util
import os
from pathlib import Path


def _load_stage17_balanced_cfg():
    cfg_path = Path(__file__).with_name(
        "habitat_dual_system_vlmap_stage17a_train_balanced_shadow_cfg.py"
    )
    spec = importlib.util.spec_from_file_location(
        "_habitat_dual_system_vlmap_stage17a_train_balanced_shadow_cfg",
        cfg_path,
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.eval_cfg


def _get_checkpoint_path():
    return os.environ.get(
        "STAGE17_PROGRESS_RANKER_CHECKPOINT",
        "checkpoints/stage17b_route_progress_v2_target_hard_balanced500_smoke/best.pt",
    )


eval_cfg = copy.deepcopy(_load_stage17_balanced_cfg())

vlmap_cfg = eval_cfg.agent.model_settings["vlmap_safety"]

# Stage17d: online shadow only.  The progress ranker scores each OccMem
# candidate event and writes hypothetical selections to memory_events.jsonl.
# It never changes S2/NextDiT actions.
vlmap_cfg["occ_memory_progress_ranker_shadow_enable"] = True
vlmap_cfg["occ_memory_progress_ranker_shadow_checkpoint"] = _get_checkpoint_path()
vlmap_cfg["occ_memory_progress_ranker_shadow_device"] = os.environ.get(
    "STAGE17_PROGRESS_RANKER_DEVICE",
    "cpu",
)
vlmap_cfg["occ_memory_progress_ranker_shadow_resilience_weight"] = float(
    os.environ.get("STAGE17_PROGRESS_RANKER_RESILIENCE_WEIGHT", "0.20")
)

run_name = os.environ.get(
    "STAGE17_BALANCED_RUN_NAME",
    "compare_vlmap_stage17d_progress_ranker_shadow_balanced",
)
output_path = f"./logs/habitat/{run_name}"
vlmap_cfg["debug_dir"] = f"{output_path}/vlmap_safety_debug"
eval_cfg.eval_settings["output_path"] = output_path
eval_cfg.eval_settings["port"] = os.environ.get("STAGE17_EVAL_PORT", "2394")
