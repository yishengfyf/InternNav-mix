from internnav.configs.agent import AgentCfg
from internnav.configs.evaluator import EnvCfg, EvalCfg

eval_cfg = EvalCfg(
    agent=AgentCfg(
        model_name='internvla_n1',
        model_settings={
            "mode": "dual_system",  # inference mode: dual_system or system2
            "model_path": "checkpoints/InternVLA-N1-DualVLN",  # path to model checkpoint
            "num_history": 8,
            "resize_w": 384,  # image resize width
            "resize_h": 384,  # image resize height
            "max_new_tokens": 1024,  # maximum number of tokens for generation
            "vis_debug": False,  # If vis_debug=True, save debug videos per episode
            "vis_debug_path": "./logs/habitat/vis_debug",
            "vlmap_safety": {
                "enable": True,
                "vlmaps_repo": "/home/yifeifeng/workspace/vlmaps",
                "grid_size": 1000,
                "cell_size": 0.05,
                "depth_scale": 1.0,
                "depth_sample_rate": 80,
                "min_depth": 0.15,
                "max_depth": 5.0,
                "obstacle_height_min": 0.15,
                "obstacle_height_max": 1.2,
                "forward_distance": 0.20,
                "turn_angle_deg": 30.0,
                "radius_cells": 0,
                "line_skip_distance": 0.10,
                "line_blocked_fraction": 0.50,
                "line_blocked_min_cells": 2,
                "line_cell_blocked_fraction": 0.25,
                "update_every_steps": 1,
                "prefer_previous_turn": True,
                "shadow_only": True,
                "debug": True,
                "debug_dir": "./logs/habitat/vlmap_safety_debug",
                "debug_log_all_events": True,
                "debug_max_snapshots": 200,
                "debug_save_on_change": True,
                "debug_save_every_steps": 20,
                "debug_crop_radius_cells": 80,
                "debug_cell_scale": 3,
                "verbose": True,
            },
        },
    ),
    env=EnvCfg(
        env_type='habitat',
        env_settings={
            # habitat sim specifications - agent, sensors, tasks, measures etc. are defined in the habitat config file
            'config_path': 'scripts/eval/configs/vln_r2r.yaml',
        },
    ),
    eval_type='habitat_vln',
    eval_settings={
        # all current parse args
        "output_path": "./logs/habitat/test_dual_system",  # output directory for logs/results
        "save_video": False,  # whether to save videos
        "epoch": 0,  # epoch number for logging
        "max_steps_per_episode": 500,  # maximum steps per episode
        # distributed settings
        "port": "2333",  # communication port
        "dist_url": "env://",  # url for distributed setup
    },
)
