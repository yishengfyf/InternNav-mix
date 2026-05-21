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
            "eval_random_seed": None,
            "eval_seed_per_episode": False,
            "eval_episode_seed_mode": "episode_index",
            "s2_prompt_conjunction_index": None,
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
                "forward_distance": 0.25,
                "turn_angle_deg": 30.0,
                "radius_cells": 0,
                "line_skip_distance": 0.08,
                # Use VLMap as a high-confidence safety layer instead of a dense local planner.
                # These thresholds reduce chair/doorframe edge false positives in Habitat.
                "line_blocked_fraction": 0.80,
                "line_blocked_min_cells": 4,
                "line_min_checked_cells": 4,
                "line_cell_blocked_fraction": 0.35,
                "update_every_steps": 1,
                "prefer_previous_turn": False,
                "repeat_block_enable": True,
                "repeat_block_count": 3,
                "repeat_block_window_steps": 10,
                "repeat_block_distance": 0.60,
                "repeat_turn_lock_steps": 2,
                "cluster_reset_distance": 0.75,
                "cluster_reset_steps": 30,
                "max_same_turn_in_cluster": 2,
                "max_replans_per_cluster": 1,
                "replan_cooldown_steps": 16,
                "recovery_turn_steps": 1,
                "stuck_block_count": 8,
                "stuck_distance": 0.15,
                "waypoint_repair_on_stuck": True,
                "max_waypoint_repairs_per_cluster": 1,
                # Only allow a small local nudge. After that, VLMap asks for replanning/logging
                # instead of continuing to override InternNav's route.
                "max_safety_changes_per_cluster": 2,
                "max_safety_changes_per_episode": 18,
                "replan_on_budget_exhaustion": True,
                "max_budget_replans_per_episode": 2,
                "action_safety_enable": True,
                "waypoint_check_enable": False,
                "waypoint_shadow_only": True,
                "waypoint_requery_enable": False,
                "waypoint_min_depth": 0.20,
                "waypoint_max_distance": 3.0,
                "waypoint_depth_patch_radius": 2,
                "waypoint_camera_pitch_deg": 30.0,
                "waypoint_source_image_width": None,
                "waypoint_source_image_height": None,
                "waypoint_save_snapshots": True,
                "waypoint_risk_threshold": 0.60,
                "waypoint_risk_min_checked_cells": 4,
                "waypoint_force_save_on_risk": True,
                "waypoint_force_save_on_block": True,
                "waypoint_force_max_snapshots": 20,
                "waypoint_requery_on_block": True,
                "waypoint_requery_on_high_risk": True,
                "waypoint_requery_risk_threshold": 0.75,
                "waypoint_requery_min_checked_cells": 20,
                "waypoint_requery_feedback_enable": True,
                "waypoint_requery_duplicate_suppression": True,
                "waypoint_requery_repeat_grid_radius": 2,
                "max_waypoint_requeries_per_episode": 1,
                "waypoint_requery_cooldown_steps": 40,
                "waypoint_recovery_enable": False,
                "waypoint_recovery_on_block": True,
                "waypoint_recovery_on_high_risk": True,
                "waypoint_recovery_risk_threshold": 0.75,
                "waypoint_recovery_min_checked_cells": 20,
                "max_waypoint_recoveries_per_episode": 2,
                "waypoint_recovery_cooldown_steps": 20,
                "waypoint_recovery_probe_distance": 0.60,
                "waypoint_recovery_require_free_probe": True,
                "waypoint_recovery_alignment_weight": 0.25,
                "waypoint_recovery_max_turn_steps": 1,
                "waypoint_recovery_candidate_angles_deg": [-30.0, 30.0],
                "shadow_only": False,
                "debug": True,
                "debug_dir": "./logs/habitat/vlmap_safety_debug",
                "debug_use_run_subdir": True,
                "debug_run_prefix": "run",
                "debug_log_all_events": True,
                "debug_max_snapshots": 60,
                "debug_save_on_change": True,
                "debug_save_every_steps": 0,
                "debug_sample_snapshots": True,
                "debug_sample_total_snapshots": 20,
                "debug_sample_images_per_episode": 2,
                "debug_sample_seed": 0,
                "debug_sample_candidate_stride": 2,
                "debug_force_on_replan": True,
                "debug_force_on_budget_suppressed": True,
                "debug_force_cluster_block_count": 3,
                "debug_force_cluster_block_interval": 3,
                "debug_force_max_snapshots": 40,
                "debug_force_max_snapshots_per_episode": 2,
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
            # Fixed evaluation slice for fair baseline/VLMap comparisons.
            # Keep these values identical across comparison runs.
            'episode_start_index': 0,
            'max_eval_episodes': 50,
            # Optional exact selection. Supports [139, 140] or [{"scene_id": "2azQ1b91cZZ", "episode_id": 139}].
            'episode_ids': None,
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
