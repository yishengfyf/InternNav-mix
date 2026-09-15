import os

from internnav.configs.agent import AgentCfg
from internnav.configs.evaluator import EnvCfg, EvalCfg


variant = os.environ.get("DUALVLN_CLOSED_LOOP_VARIANT", "B0")
if variant not in {"B0", "B1", "B2", "M1", "M2Z", "M2"}:
    raise ValueError(f"unsupported closed-loop variant: {variant}")
adapter_path = os.environ.get("DUALVLN_EVIDENCE_ADAPTER") if variant != "B0" else None
if variant != "B0" and not adapter_path:
    raise ValueError(f"{variant} closed-loop evaluation requires DUALVLN_EVIDENCE_ADAPTER")
evidence_ablation = {
    "B1": "null",
    "B2": "content",
    "M1": "spatial",
    "M2Z": "spatial",
    "M2": "task_spatial",
}.get(variant)
normalize_task_state = os.environ.get("DUALVLN_NORMALIZE_TASK_STATE", "0") == "1"
latent_query_bypass = os.environ.get("DUALVLN_LATENT_QUERY_BYPASS", "1") == "1"
num_history = int(os.environ.get("DUALVLN_NUM_HISTORY", "2"))

eval_cfg = EvalCfg(
    agent=AgentCfg(
        model_name="internvla_n1",
        model_settings={
            "mode": "dual_system",
            "model_path": "/home/yifeifeng/workspace/InternNav/checkpoints/InternVLA-N1",
            "evidence_adapter_path": adapter_path,
            "evidence_ablation": evidence_ablation,
            "evidence_normalize_task_state": normalize_task_state,
            "evidence_latent_query_bypass": latent_query_bypass,
            "num_history": num_history,
            "resize_w": 384,
            "resize_h": 384,
            "max_new_tokens": 128,
            "vis_debug": False,
        },
    ),
    env=EnvCfg(
        env_type="habitat",
        env_settings={
            "config_path": "scripts/dualvln_mainline/configs/habitat_r2r_short_smoke.yaml",
            "max_episodes": int(os.environ.get("DUALVLN_MAX_EPISODES", "1")),
        },
    ),
    eval_type="habitat_vln",
    eval_settings={
        "output_path": os.environ["DUALVLN_CLOSED_LOOP_OUTPUT"],
        "save_video": False,
        "epoch": 0,
        "max_steps_per_episode": int(os.environ.get("DUALVLN_MAX_STEPS", "24")),
        "protocol_name": "r2r_val_unseen_sorted_ideal_pose_short_smoke_v1",
        "seed": int(os.environ.get("DUALVLN_CLOSED_LOOP_SEED", "23")),
        "port": os.environ.get("DUALVLN_DIST_PORT", "2347"),
        "dist_url": "env://",
    },
)
