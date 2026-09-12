import os

from internnav.configs.agent import AgentCfg
from internnav.configs.evaluator import EnvCfg, EvalCfg


variant = os.environ.get("DUALVLN_CLOSED_LOOP_VARIANT", "B0")
if variant not in {"B0", "M2"}:
    raise ValueError(f"unsupported closed-loop variant: {variant}")
adapter_path = os.environ.get("DUALVLN_EVIDENCE_ADAPTER") if variant == "M2" else None
if variant == "M2" and not adapter_path:
    raise ValueError("M2 closed-loop evaluation requires DUALVLN_EVIDENCE_ADAPTER")

eval_cfg = EvalCfg(
    agent=AgentCfg(
        model_name="internvla_n1",
        model_settings={
            "mode": "dual_system",
            "model_path": "/home/yifeifeng/workspace/InternNav/checkpoints/InternVLA-N1",
            "evidence_adapter_path": adapter_path,
            "num_history": 2,
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
        "port": "2347",
        "dist_url": "env://",
    },
)
