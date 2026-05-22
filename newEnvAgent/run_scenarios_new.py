"""
run_scenarios_new.py

Trial-script for diffusion-guided kinodynamic RRT using the new decoupled framework.
Modelled after MAPF trial_scripts (e.g. trial_script_rrt_SOC.py):
  obstacles hardcoded inline, env + agent created directly, planner run immediately.
  No CSV reading, no benchmark loops.
"""

import os
import sys
import time
import random

import numpy as np
import torch
from scipy.spatial import KDTree
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from train_diffusion_policy import init_noise_pred_net
from policies.fm_policy import DiffusionSampler

sys.path.insert(0, os.path.dirname(__file__))
from environments import RectangularEnvironment, CircularObstacle2D, RectangleObstacle2D
from car import BicycleCar


# ---------------------------------------------------------------------------
# Local map from continuous environment
# ---------------------------------------------------------------------------

def create_local_map_from_env(env, agent_x, agent_y, agent_yaw, map_size=20, scale=0.2):
    """
    Build an (N, N) binary occupancy grid centred on the agent in its local frame.

    Parameters
    ----------
    env                       : RectangularEnvironment
    agent_x, agent_y, agent_yaw : current robot pose
    map_size                  : grid side length N (cells)
    scale                     : metres per cell

    Returns
    -------
    local_map : np.ndarray (N, N), dtype float32
        0.0 = free, 1.0 = occupied / out-of-bounds
    """
    N = int(map_size)
    half = (N * scale) / 2.0
    coords = np.linspace(-half + scale / 2.0, half - scale / 2.0, N)
    x_local, y_local = np.meshgrid(coords, coords)

    cos_h = np.cos(agent_yaw)
    sin_h = np.sin(agent_yaw)
    x_world = cos_h * x_local - sin_h * y_local + agent_x
    y_world = sin_h * x_local + cos_h * y_local + agent_y

    local_map = np.zeros((N, N), dtype=np.float32)

    oob = ((x_world < env.env_start[0]) | (x_world > env.env_start[0] + env.size[0]) |
           (y_world < env.env_start[1]) | (y_world > env.env_start[1] + env.size[1]))
    local_map[oob] = 1.0

    for obs in env.obstacles:
        if isinstance(obs, CircularObstacle2D):
            dx = x_world - obs.x
            dy = y_world - obs.y
            local_map[dx * dx + dy * dy < obs.r * obs.r] = 1.0
        elif isinstance(obs, RectangleObstacle2D):
            hw, hh = obs.w / 2.0, obs.h / 2.0
            inside = ((x_world >= obs.x - hw) & (x_world <= obs.x + hw) &
                      (y_world >= obs.y - hh) & (y_world <= obs.y + hh))
            local_map[inside] = 1.0

    return local_map


# ---------------------------------------------------------------------------
# RRT node
# ---------------------------------------------------------------------------

class Node:
    def __init__(self, state, parent_action_seq=None, parent_states_seq=None, parent=None):
        self.state = state
        self.parent_action_seq = parent_action_seq
        self.parent_states_seq = parent_states_seq
        self.parent = parent
        self.num_visit = 0


# ---------------------------------------------------------------------------
# RRT planner
# ---------------------------------------------------------------------------

class NewRRTPlanner:
    """
    Kinodynamic RRT using BicycleCar + RectangularEnvironment directly.
    No gym interface. Local map built from geometric env via create_local_map_from_env().
    """

    def __init__(self, start, goal, agent, env, sampler,
                 action_horizon=8,
                 prop_duration=None,
                 goal_sample_rate=0.15,
                 goal_conditioning_bias=0.85,
                 goal_radius=0.5,
                 local_map_size=20,
                 local_map_scale=0.2,
                 time_budget=120,
                 verbose=False):

        self.agent = agent
        self.env = env
        self.sampler = sampler

        self.start = np.array(start, dtype=np.float64)
        self.goal = np.array(goal, dtype=np.float64)
        self.goal_radius = goal_radius

        self.action_horizon = action_horizon
        self.prop_duration_schedule = prop_duration if prop_duration is not None else [64]
        self.goal_sample_rate = goal_sample_rate
        self.goal_conditioning_bias = goal_conditioning_bias

        self.local_map_size = local_map_size
        self.local_map_scale = local_map_scale
        self.time_budget = time_budget
        self.verbose = verbose

        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

        start_node = Node(self.start)
        self.node_list = [start_node]
        self.kd_tree = KDTree([self.start[:2]])

        self.results = {
            "iterations": 0, "time": 0,
            "path": None, "actions": None,
            "number_of_nodes": 0, "path_time": 0,
        }

    def reset(self):
        start_node = Node(self.start)
        self.node_list = [start_node]
        self.kd_tree = KDTree([self.start[:2]])
        self.results = {
            "iterations": 0, "time": 0,
            "path": None, "actions": None,
            "number_of_nodes": 0, "path_time": 0,
        }

    def _random_sample(self):
        if random.random() > self.goal_sample_rate:
            return self.agent.get_random_point(self.env)[None, :]
        sample = np.zeros((1, self.agent.state_length), dtype=np.float64)
        sample[0] = self.goal
        return sample

    def _nearest_node(self, sample):
        _, idx = self.kd_tree.query(sample[:, :2], k=1)
        return self.node_list[idx[0]]

    def _propagate_chunk(self, start_state, action_sequence):
        """
        Execute action_horizon steps of the bicycle model directly.
        Returns (final_state, done, trimmed_actions, states_seq).
          done = True  → goal reached
          done = None  → collision
          done = False → chunk complete, neither
        """
        state = np.array(start_state, dtype=np.float64)
        states_seq = np.zeros((self.action_horizon + 1, self.agent.state_length), dtype=np.float64)
        states_seq[0] = state

        circ_obs  = self.env.static_circular_obstacles
        rect_obs  = self.env.static_rectangular_obstacles
        env_start = self.env.env_start
        env_size  = self.env.size
        ob_buf    = self.env.obstacle_buffer
        bb_buf    = self.env.boundary_buffer
        radius    = self.agent.radius

        for i in range(self.action_horizon):
            action = np.array(action_sequence[i], dtype=np.float64)
            next_state, _ = self.agent.get_next_state(state, action, self.agent.dt, num_steps=1)

            if not BicycleCar.is_state_valid(next_state, radius, env_start, env_size,
                                              circ_obs, rect_obs, ob_buf, bb_buf):
                return next_state, None, action_sequence[:i], states_seq[:i][None, :]

            state = next_state
            states_seq[i + 1] = state

            if BicycleCar.agent_reached_goal(state, self.goal, self.goal_radius):
                return state, True, action_sequence[:i + 1], states_seq[:i + 1][None, :]

        return state, False, action_sequence[:self.action_horizon], states_seq[:self.action_horizon][None, :]

    def _generate_path(self, goal_node):
        path_list = []
        actions_list = []
        node = goal_node
        while node is not None:
            path_list = [node.state] + path_list
            if node.parent_states_seq is not None:
                path_list = list(node.parent_states_seq[0]) + path_list
            if node.parent_action_seq is not None:
                actions_list = list(node.parent_action_seq) + actions_list
            node = node.parent
        path    = np.array(path_list,    dtype=np.float32) if path_list    else None
        actions = np.array(actions_list, dtype=np.float32) if actions_list else None
        return path, actions

    def plan(self):
        start_time = time.time()
        curr_time  = start_time
        i = 0

        while (curr_time - start_time) < self.time_budget:
            if self.verbose:
                print(f"\rIteration {i}, elapsed {curr_time - start_time:.1f}s", end="")

            sample    = self._random_sample()
            curr_node = self._nearest_node(sample)
            curr_state = curr_node.state.copy()

            edge_length = (self.prop_duration_schedule[curr_node.num_visit]
                           if curr_node.num_visit < len(self.prop_duration_schedule)
                           else self.prop_duration_schedule[-1])
            curr_node.num_visit += 1

            goal_for_sampler = (sample[0, :2] if random.random() > self.goal_conditioning_bias
                                else self.goal[:2])

            full_action_seq = None
            full_states_seq = None
            done = False
            prev_actions = curr_node.parent_action_seq
            prev_states  = (curr_state[None, None, :]
                            if curr_node.parent_states_seq is None
                            else curr_node.parent_states_seq)

            for _ in range(edge_length // self.action_horizon):
                local_map = create_local_map_from_env(
                    self.env,
                    curr_state[0], curr_state[1], curr_state[2],
                    self.local_map_size, self.local_map_scale)
                local_map_t = torch.tensor(local_map).to(self.device)

                sampled_actions = self.sampler(
                    prev_states, prev_actions=prev_actions,
                    goal=goal_for_sampler, local_map=local_map_t)[0]

                curr_state, done, curr_action_seq, curr_states_seq = \
                    self._propagate_chunk(curr_state, sampled_actions)

                full_action_seq = (np.concatenate((full_action_seq, curr_action_seq))
                                   if full_action_seq is not None else curr_action_seq)
                full_states_seq = (np.concatenate((full_states_seq, curr_states_seq), axis=1)
                                   if full_states_seq is not None else curr_states_seq)
                prev_actions = curr_action_seq
                prev_states  = curr_states_seq

                if done is None:
                    curr_state = None
                    break
                if done:
                    break

            if curr_state is not None:
                new_node = Node(curr_state, full_action_seq, full_states_seq, parent=curr_node)
                self.node_list.append(new_node)
                self.kd_tree = KDTree([n.state[:2] for n in self.node_list])

            if done:
                self.results["iterations"]     = i
                self.results["time"]            = time.time() - start_time
                self.results["number_of_nodes"] = len(self.node_list)
                path, actions = self._generate_path(new_node)
                self.results["path"]      = path
                self.results["actions"]   = actions
                self.results["path_time"] = (len(path) * self.agent.dt if path is not None else 0)
                if self.verbose:
                    print(f"\n  Goal reached in {i} iterations.")
                return path, actions

            i += 1
            curr_time = time.time()

        self.results["iterations"]     = i
        self.results["time"]           = time.time() - start_time
        self.results["number_of_nodes"] = len(self.node_list)
        if self.verbose:
            print(f"\n  No path found in {i} iterations.")
        return None, None


# ---------------------------------------------------------------------------
# Model loader
# ---------------------------------------------------------------------------

def build_sampler(cfg_file='carmaze', checkpoint_path='checkpoints/carmaze.pt'):
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
# Trial script
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    obstacles = [
        CircularObstacle2D(10, 10, 2),
        CircularObstacle2D(16, 25, 3),
        CircularObstacle2D(20, 5,  2),
        CircularObstacle2D(25, 15, 4),
        RectangleObstacle2D(30, 20, 6, 4),
    ]

    env   = RectangularEnvironment(40, 40, obstacles)
    agent = BicycleCar(agent_id=0)

    start = np.array([2.0,  2.0,  0.0, 0.0, 0.0, 0.0], dtype=np.float64)
    goal  = np.array([35.0, 35.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float64)

    sampler, cfg = build_sampler(cfg_file='carmaze', checkpoint_path='checkpoints/carmaze.pt')

    planner = NewRRTPlanner(
        start=start, goal=goal,
        agent=agent, env=env, sampler=sampler,
        action_horizon=cfg.get('action_horizon', 8),
        prop_duration=cfg.get('prop_duration', [64]),
        goal_conditioning_bias=cfg.get('goal_conditioning_bias', 0.85),
        local_map_size=cfg.get('local_map_size', 20),
        local_map_scale=cfg.get('local_map_scale', 0.2),
        goal_radius=0.5,
        time_budget=120,
        verbose=True,
    )

    path, actions = planner.plan()

    if path is not None:
        traj_len = float(np.sum(np.linalg.norm(np.diff(path[:, :2], axis=0), axis=1)))
        print(f"Path found: {len(path)} states, length {traj_len:.2f} m, "
              f"{planner.results['time']:.2f}s, "
              f"{planner.results['number_of_nodes']} nodes.")
    else:
        print(f"No path found in {planner.results['time']:.2f}s.")
