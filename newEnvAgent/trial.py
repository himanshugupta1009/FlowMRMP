"""
trial.py

Hardcoded environment, 10 RRT trials, 30-second budget each.
Prints per-trial stats and a summary at the end.
"""

import os
import sys
import numpy as np
import torch
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from train_diffusion_policy import init_noise_pred_net
from policies.fm_policy import DiffusionSampler

sys.path.insert(0, os.path.dirname(__file__))
from environments import RectangularEnvironment, CircularObstacle2D, RectangleObstacle2D
from car import BicycleCar
from new_RRT import NewRRTPlanner


# ---------------------------------------------------------------------------
# Model loader
# ---------------------------------------------------------------------------

def build_sampler(cfg_file='carmaze', checkpoint_path='checkpoints/carmaze_step_40000.pt'):
    import yaml

    unet_dims = {
        'small':  [64,   128,  256],
        'medium': [256,  512,  1024],
        'large':  [512,  1024, 2048],
        'xlarge': [1024, 2048, 4096],
    }

    cfg_path = os.path.join(os.path.dirname(__file__), '..', 'cfgs', f'{cfg_file}.yaml')
    with open(cfg_path) as f:
        cfg = yaml.safe_load(f)

    obs_history             = cfg.get('obs_history', 1)
    action_history          = cfg.get('action_history', 1)
    goal_conditioned        = cfg.get('goal_conditioned', True)
    local_map_conditioned   = cfg.get('local_map_conditioned', True)
    local_map_size          = cfg.get('local_map_size', 20)
    local_map_embedding_dim = cfg.get('local_map_embedding_dim', 400)
    num_diffusion_iters     = cfg.get('planning_diffusion_iters', 1)
    unet_down_dims          = unet_dims[cfg.get('denoiser_size', 'large')]
    pred_horizon            = cfg.get('pred_horizon', 64)
    action_dim              = 2
    obs_dim                 = 3
    goal_dim                = 2
    policy                  = cfg.get('policy', 'flow_matching')
    env_id                  = cfg.get('env_id', 'carmaze')

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    noise_scheduler = DDPMScheduler(
        num_train_timesteps=num_diffusion_iters,
        beta_schedule='squaredcos_cap_v2',
        clip_sample=True,
        prediction_type='epsilon')

    noise_pred_net = init_noise_pred_net(
        input_dim=action_dim,
        action_dim=action_dim,
        obs_dim=obs_dim,
        obs_history=obs_history,
        action_history=action_history,
        goal_conditioned=goal_conditioned,
        goal_dim=goal_dim,
        local_map_conditioned=local_map_conditioned,
        local_map_encoder='resnet',
        local_map_embedding_dim=local_map_embedding_dim,
        local_map_size=local_map_size,
        down_dims=unet_down_dims,
    )

    ckpt_full = os.path.join(os.path.dirname(__file__), '..', checkpoint_path)
    checkpoint = torch.load(ckpt_full, map_location=device)
    noise_pred_net.load_state_dict(checkpoint['noise_pred_net_state_dict'])
    noise_pred_net = noise_pred_net.to(device).eval()

    sampler = DiffusionSampler(
        noise_pred_net, noise_scheduler, env_id,
        policy=policy,
        pred_horizon=pred_horizon,
        action_dim=action_dim,
        prediction_type='actions',
        obs_history=obs_history,
        action_history=action_history,
        goal_conditioned=goal_conditioned,
        num_diffusion_iters=num_diffusion_iters,
        local_map_size=local_map_size,
    ).eval()

    return sampler, cfg


# ---------------------------------------------------------------------------
# Trial runner
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    NUM_TRIALS  = 10
    TIME_BUDGET = 30    # seconds per trial

    # ------------------------------------------------------------------
    # Environment  (40 x 40 m)
    # ------------------------------------------------------------------
    obstacles = [
        CircularObstacle2D(10, 10, 2),
        CircularObstacle2D(20, 8,  2.5),
        CircularObstacle2D(15, 22, 3),
        CircularObstacle2D(28, 18, 2),
        CircularObstacle2D(25, 32, 2.5),
        RectangleObstacle2D(10, 30, 6, 3),
        RectangleObstacle2D(30, 10, 4, 5),
    ]

    env   = RectangularEnvironment(40, 40, obstacles)
    agent = BicycleCar(agent_id=0)

    start = np.array([2.0,  2.0,  0.0, 0.0, 0.0, 0.0], dtype=np.float64)
    goal  = np.array([37.0, 37.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float64)

    # ------------------------------------------------------------------
    # Model
    # ------------------------------------------------------------------
    sampler, cfg = build_sampler(cfg_file='carmaze', checkpoint_path='checkpoints/carmaze_step_40000.pt')

    planner = NewRRTPlanner(
        start=start, goal=goal,
        agent=agent, env=env, sampler=sampler,
        action_horizon=cfg.get('action_horizon', 8),
        prop_duration=cfg.get('prop_duration', [64]),
        goal_conditioning_bias=cfg.get('goal_conditioning_bias', 0.85),
        local_map_size=cfg.get('local_map_size', 20),
        local_map_scale=cfg.get('local_map_scale', 0.2),
        goal_radius=0.5,
        time_budget=TIME_BUDGET,
        verbose=True,
    )

    # ------------------------------------------------------------------
    # Header
    # ------------------------------------------------------------------
    print("=" * 60)
    print(f"  Environment : 40 x 40 m, {len(obstacles)} obstacles")
    print(f"  Start       : {start[:2]}")
    print(f"  Goal        : {goal[:2]}")
    print(f"  Trials      : {NUM_TRIALS}   Time budget : {TIME_BUDGET}s each")
    print("=" * 60)

    # ------------------------------------------------------------------
    # Run trials
    # ------------------------------------------------------------------
    records = []

    for trial in range(1, NUM_TRIALS + 1):
        print(f"\n[Trial {trial}/{NUM_TRIALS}]")
        planner.reset()
        path, _ = planner.plan()
        r = planner.results

        success  = path is not None
        traj_len = (float(np.sum(np.linalg.norm(np.diff(path[:, :2], axis=0), axis=1)))
                    if success else None)

        records.append({
            "success":    success,
            "time":       r["time"],
            "iterations": r["iterations"],
            "nodes":      r["number_of_nodes"],
            "traj_len":   traj_len,
        })

        status = "PASS" if success else "FAIL"
        length_str = f"{traj_len:.2f} m" if traj_len is not None else "—"
        print(f"  {status}  |  time {r['time']:.1f}s  |  "
              f"iters {r['iterations']}  |  nodes {r['number_of_nodes']}  |  "
              f"path length {length_str}")

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    passed   = [rec for rec in records if rec["success"]]
    failed   = [rec for rec in records if not rec["success"]]
    n_pass   = len(passed)
    n_fail   = len(failed)

    print("\n" + "=" * 60)
    print(f"  RESULTS : {n_pass} passed,  {n_fail} failed  ({n_pass}/{NUM_TRIALS})")
    print("-" * 60)

    avg_time  = np.mean([r["time"]       for r in records])
    avg_iters = np.mean([r["iterations"] for r in records])
    avg_nodes = np.mean([r["nodes"]      for r in records])
    print(f"  Avg time       : {avg_time:.1f}s")
    print(f"  Avg iterations : {avg_iters:.0f}")
    print(f"  Avg nodes      : {avg_nodes:.0f}")

    if passed:
        avg_len = np.mean([r["traj_len"] for r in passed])
        print(f"  Avg path length: {avg_len:.2f} m  (successful trials only)")

    print("=" * 60)
