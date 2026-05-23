#edge_bundle.py

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
import time
from utils import get_dtype_from_input,find_roundoff_decimal_digits

"""
Vehicle State S = (x,y,theta,v,omega)
Vehicle Control U = (acceleration, steering_angle)

Stored Edge bundle is a dictionary with the following keys
    1) 'actions' : array of actions where every action a_i is a ndarray of values.
    2) 'timesteps': array of time steps where a time step t_i denotes how long the action was executed for
    2) 'final_states' : array of the final state of an agent after the action a_i was executed for time step t_i 
"""

class EdgeBundle:
    def __init__(self,data,*,
                 use_all_edges=False,
                 fix_num_edges=1000,
                 rng_seed=23):

        if use_all_edges:
            fix_num_edges = len(data['timesteps'])
            self.chosen_indices = np.arange(fix_num_edges)
        else:
            self.rng = np.random.default_rng(rng_seed)
            total_edges = len(data['timesteps'])
            self.chosen_indices = self.rng.choice(total_edges, size=fix_num_edges, replace=False)

        self.num_edges = fix_num_edges
        # self._fields = list(data.keys())
        self._fields = ['start_states', 'actions', 'timesteps', 'trajectories', 'trajectory_lengths']
        for field in self._fields:
            setattr(self, field, data[field][self.chosen_indices])
        self.edge_index = 0
        self.distance_from_random_point = np.zeros(self.num_edges)

        if 'final_states' not in self._fields:
            self.final_states = []
            for i in range(self.num_edges):
                trajectory = self.trajectories[i]
                trajectory_length = self.trajectory_lengths[i]
                final_state = trajectory[trajectory_length - 1]
                self.final_states.append(final_state)
            self.final_states = np.array(self.final_states)
        
    def get_next_edge(self):
        if self.edge_index < self.num_edges:
            action = self.actions[self.edge_index]
            timestep = self.timesteps[self.edge_index]
            final_state = self.final_states[self.edge_index]
            self.edge_index += 1
            return action, timestep, final_state
        else:
            return None, None, None
        
    def get_edge(self, edge_index):
        if edge_index < self.num_edges:
            action = self.actions[edge_index]
            timestep = self.timesteps[edge_index]
            final_state = self.final_states[edge_index]
            return action, timestep, final_state
        else:
            print("Edge index out of bounds")
            return None, None, None
        
    def plot_edges_2d(self, edge_ids=None, show_starts=True):
        """
        Plot edges in 2D using the stored trajectories.

        Parameters
        ----------
        edge_ids : array-like of int, optional
            Indices of edges in this EdgeBundle (0 .. self.num_edges-1).
            If None, all edges in the bundle are plotted.
        show_starts : bool, default True
            Whether to scatter-plot the start states for the selected edges.
        """
        trajs = self.trajectories          # shape: (num_edges, max_len, state_dim)
        lengths = self.trajectory_lengths  # shape: (num_edges,)

        E = self.num_edges

        if edge_ids is None:
            edge_ids = np.arange(E, dtype=int)
        else:
            edge_ids = np.asarray(edge_ids, dtype=int)

        # Filter out-of-range indices
        valid_mask = (edge_ids >= 0) & (edge_ids < E)
        if not np.all(valid_mask):
            print("Warning: some edge_ids are out of range and will be ignored.")
            edge_ids = edge_ids[valid_mask]

        if edge_ids.size == 0:
            print("No valid edge IDs to plot.")
            return

        segments = []
        for i in edge_ids:
            L = int(lengths[i])
            xy = trajs[i, :L, :2]          # take x,y of the trajectory
            if np.isnan(xy).any():
                xy = xy[np.isfinite(xy).all(axis=1)]
            if xy.shape[0] >= 2:
                segments.append(xy)

        if not segments:
            print("No valid segments to plot (all NaN/too short).")
            return

        fig, ax = plt.subplots(figsize=(7, 7))
        lc = LineCollection(segments, linewidths=0.8)
        ax.add_collection(lc)

        all_xy = np.vstack(segments)
        ax.set_xlim(all_xy[:, 0].min(), all_xy[:, 0].max())
        ax.set_ylim(all_xy[:, 1].min(), all_xy[:, 1].max())

        ax.set_aspect("equal", adjustable="box")
        ax.set_title(f"Edge bundle (showing {len(segments)} edges)")
        ax.set_xlabel("x")
        ax.set_ylabel("y")

        # Optional: plot start states if present
        if show_starts and hasattr(self, "start_states"):
            starts = np.asarray(self.start_states)
            ax.scatter(starts[edge_ids, 0], starts[edge_ids, 1], s=6, c="red", alpha=0.5)

        plt.show()

    def plot_edges_by_ids(self, edge_ids, show_starts=True):
        """
        Convenience wrapper to plot only the given edge IDs.

        Parameters
        ----------
        edge_ids : array-like of int
            Indices in this EdgeBundle (0 .. self.num_edges-1).
        """
        self.plot_edges_2d(edge_ids=edge_ids, show_starts=show_starts)



class EdgeBundleTraj(EdgeBundle):
    def __init__(self, data, fix_num_edges=1000):
        super().__init__(data, fix_num_edges)
        self.trajectories = data['trajectories'][:fix_num_edges]

    def get_trajectory(self, traj_index):
        if traj_index < self.num_edges:
            return self.trajectories[traj_index]
        else:
            return None

# Load the edge bundle
# edge_bundle_file_location = 'edge_bundles_npz/eb_unicycle_edges_100000.npz' 
# data = np.load(edge_bundle_file_location)
# d = EdgeBundle(data)


"""
Step to run this file or anything in the Python REPL:
    import sys
    sys.path.append('./src')
    from edge_bundle import * 
    # from edge_bundle import EdgeBundle
    # import edge_bundle as eb

    edge_bundle_file_location = 'edge_bundles_npz/eb_unicycle_edges_100000.npz' 
    data = np.load(edge_bundle_file_location)
    d = EdgeBundle(data)
    action, timestep, final_state = d.get_next_edge()
    print(action, timestep, final_state)
"""


class GenerateEdgeBundle:
    def __init__(self,*, 
                env, agent, 
                get_start_state, 
                num_edges=100000,
                minimum_time_step=0.1, 
                max_sample_T=1.5,
                rng_seed=11,
                ):
        self.env = env
        self.agent = agent
        self.num_edges = num_edges
        self.get_start_state_function = get_start_state
        #Smallest time duration between consecutive agent positions
        self.minimum_time_step = minimum_time_step
        #Maximum time duration for which an action can be executed
        self.max_sample_T = max_sample_T
        self.rng_seed = rng_seed
        self.rng = np.random.default_rng(self.rng_seed)
        #Set number of roundoff digits for time sampling
        self.roundoff_digits = find_roundoff_decimal_digits(self.minimum_time_step)
        self.edge_bundle = {}
        self.num_attempted_edges = 0
        self.num_rejected_edges = 0
        self.num_accepted_edges = 0

    def get_random_time(self):
        return round(self.rng.uniform(self.minimum_time_step,
                            self.max_sample_T), self.roundoff_digits)

    def path_respects_dynamic_limits(self, path):
        """
        Return True when every state in the path respects the agent's configured
        dynamic limits. Agents without explicit dynamic-limit metadata are
        treated as always valid.
        """
        if not hasattr(self.agent, "dynamic_limit_indices"):
            return True
        if not hasattr(self.agent, "dynamic_limit_values"):
            return True

        limit_indices = np.asarray(self.agent.dynamic_limit_indices, dtype=np.int64)
        limit_values = np.asarray(self.agent.dynamic_limit_values, dtype=np.float64)

        if limit_indices.shape[0] == 0:
            return True

        for state in path:
            for k in range(limit_indices.shape[0]):
                idx = int(limit_indices[k])
                if abs(state[idx]) > limit_values[k]:
                    return False
        return True

    def generate_edge_bundle(self):
        t0 = time.time()
    
        start_states_array = []
        actions_array = []
        timesteps_array = []
        trajectory_array = []
        trajectory_length_array = []
        self.num_attempted_edges = 0
        self.num_rejected_edges = 0
        self.num_accepted_edges = 0

        while len(start_states_array) < self.num_edges:
            self.num_attempted_edges += 1
            start_state = self.get_start_state_function(self.env,self.agent,self.rng)
            action = self.agent.get_random_action(self.rng)
            timestep = self.get_random_time()
            num_record_steps = round(timestep/self.minimum_time_step)
            final_state, _path = self.agent.get_next_state(start_state,
                                    action, timestep, num_record_steps)

            if not self.path_respects_dynamic_limits(_path):
                self.num_rejected_edges += 1
                continue

            start_states_array.append(start_state)
            actions_array.append(action)
            timesteps_array.append(timestep)
            trajectory_array.append(_path)
            trajectory_length_array.append(len(_path))
            self.num_accepted_edges += 1


        np_start_states = np.array(start_states_array)
        np_actions = np.array(actions_array)
        np_timesteps = np.array(timesteps_array)
        np_trajectory_lengths = np.array(trajectory_length_array)

        max_trajectory_length = round(self.max_sample_T/self.minimum_time_step)
        state_length = len(start_states_array[0])
        # np_trajectory_array = np.zeros((self.num_edges, max_trajectory_length, state_length))
        np_trajectory_array = np.full((self.num_edges, max_trajectory_length, state_length), 
                                    np.nan, dtype=np.float64)
        for i in range(self.num_edges):
            np_trajectory_array[i,:np_trajectory_lengths[i],:] = np.array(trajectory_array[i])

        edge_bundle = {
                    'start_states':np.array(np_start_states),
                    'actions':np.array(np_actions),
                    'timesteps':np.array(np_timesteps),
                    'trajectories':np.array(np_trajectory_array),
                    'trajectory_lengths':np.array(np_trajectory_lengths),
                    'generation_attempts': np.array([self.num_attempted_edges], dtype=np.int64),
                    'generation_rejections': np.array([self.num_rejected_edges], dtype=np.int64),
                    'generation_acceptances': np.array([self.num_accepted_edges], dtype=np.int64),
                    'generation_time_sec': np.array([time.time() - t0], dtype=np.float64),
                }
        
        self.edge_bundle = edge_bundle
        print(f"Generated {self.num_accepted_edges} valid edges after {self.num_attempted_edges} attempts in {edge_bundle['generation_time_sec'][0]:.3f} s")
        return edge_bundle

    def plot_length_hist(self, edge_bundle=None):
        if edge_bundle is None:
            edge_bundle = self.edge_bundle

        bins = np.arange(0, round(self.max_sample_T/self.minimum_time_step)+1, 1)-0.5
        lengths = np.asarray(edge_bundle["trajectory_lengths"])
        plt.figure(figsize=(6,4))
        plt.hist(lengths, bins=bins)
        plt.title("Trajectory length distribution")
        plt.xlabel("length (steps)")
        plt.ylabel("count")
        plt.show()

    def plot_edges_2d(self, edge_bundle=None, max_edges=2000, stride=1, show_starts=True):
        if edge_bundle is None:
            edge_bundle = self.edge_bundle

        trajs = edge_bundle["trajectories"]
        lengths = edge_bundle["trajectory_lengths"]
        E = trajs.shape[0]
        idx = np.arange(0, E, stride)[:max_edges]

        segments = []
        for i in idx:
            xy = trajs[i, :int(lengths[i]), :2]
            if np.isnan(xy).any():
                xy = xy[np.isfinite(xy).all(axis=1)]
            if xy.shape[0] >= 2:
                segments.append(xy)

        fig, ax = plt.subplots(figsize=(7, 7))
        lc = LineCollection(segments, linewidths=0.8)
        ax.add_collection(lc)

        if segments:
            all_xy = np.vstack(segments)
            ax.set_xlim(all_xy[:, 0].min(), all_xy[:, 0].max())
            ax.set_ylim(all_xy[:, 1].min(), all_xy[:, 1].max())

        ax.set_aspect("equal", adjustable="box")
        ax.set_title(f"Edge bundle (showing {len(segments)} edges)")
        ax.set_xlabel("x")
        ax.set_ylabel("y")

        if show_starts and "start_states" in edge_bundle:
            starts = np.asarray(edge_bundle["start_states"])
            ax.scatter(starts[idx, 0], starts[idx, 1], s=6, c="red", alpha=0.5)

        plt.show()

    def plot_edges_by_ids(self, edge_ids, edge_bundle=None, show_starts=True):
        """
        Plot specific edges from the edge bundle given their indices.

        Parameters
        ----------
        edge_ids : array-like of int
            Indices of edges to plot (0-based).
        edge_bundle : dict, optional
            Edge bundle dict. If None, uses self.edge_bundle.
        show_starts : bool, default True
            Whether to scatter-plot the start states for the selected edges.
        """
        if edge_bundle is None:
            edge_bundle = self.edge_bundle

        trajs = edge_bundle["trajectories"]
        lengths = edge_bundle["trajectory_lengths"]
        E = trajs.shape[0]

        # Convert to NumPy array and filter invalid ids
        edge_ids = np.asarray(edge_ids, dtype=int)
        valid_mask = (edge_ids >= 0) & (edge_ids < E)
        if not np.all(valid_mask):
            print("Warning: some edge_ids are out of range and will be ignored.")
            edge_ids = edge_ids[valid_mask]

        if edge_ids.size == 0:
            print("No valid edge IDs to plot.")
            return

        segments = []
        for i in edge_ids:
            L = int(lengths[i])
            xy = trajs[i, :L, :2]
            if np.isnan(xy).any():
                xy = xy[np.isfinite(xy).all(axis=1)]
            if xy.shape[0] >= 2:
                segments.append(xy)

        if not segments:
            print("No valid segments to plot (all NaN/too short).")
            return

        fig, ax = plt.subplots(figsize=(7, 7))
        lc = LineCollection(segments, linewidths=0.8)
        ax.add_collection(lc)

        all_xy = np.vstack(segments)
        ax.set_xlim(all_xy[:, 0].min(), all_xy[:, 0].max())
        ax.set_ylim(all_xy[:, 1].min(), all_xy[:, 1].max())

        ax.set_aspect("equal", adjustable="box")
        ax.set_title(f"Edge bundle (showing {len(segments)} edges)")
        ax.set_xlabel("x")
        ax.set_ylabel("y")

        if show_starts and "start_states" in edge_bundle:
            starts = np.asarray(edge_bundle["start_states"])
            ax.scatter(starts[edge_ids, 0], starts[edge_ids, 1], s=6, c="red", alpha=0.5)

        plt.show()


"""
Code to read db-A* motion primitives

import msgpack
filename = '/home/himanshu/Downloads/unicycle1_v0__ispso__2023_04_03__14_56_57.bin.im.bin.im.bin.msgpack'
with open(filename, 'rb') as f:
    data = msgpack.load(f)

#data is a dictionary
#data['data'] is a list of motion primitives
data['data'][0]

for i in range(10000):
    print(data['data'][i]['states'][0])

    


"""












"""
#Unicycle: Generate an edge bundle and save it 

import sys
sys.path.append('./src')
import numpy as np
import matplotlib.pyplot as plt
from Environments import SquareEnvironment, CircularObstacle2D
from Agents import UniCycle
from edge_bundle import GenerateEdgeBundle
from mapf_env_square_agent_unicycle import get_unicycle_agent

obstacles = []
env = SquareEnvironment(40, 40, obstacles)
agent = get_unicycle_agent(1)

def sample_with_extremes(rng, low, high, p_extreme=0.1):
    if rng.random() < p_extreme:
        # pick exactly low or high
        return low if rng.random() < 0.5 else high
    else:
        return rng.uniform(low, high)

def get_start_state_kinodynamic_TI_unicycle(env, agent, rng):
    theta = sample_with_extremes(rng, 0.0, 2*np.pi, p_extreme=0.1)
    start_state = np.array([0.0, 0.0, theta])
    return start_state

from edge_bundle import GenerateEdgeBundle
gen_eb = GenerateEdgeBundle(env = env, 
                            agent=agent, 
                            num_edges=100000,
                            get_start_state=get_start_state_kinodynamic_TI_unicycle,
                            minimum_time_step=0.1,
                            max_sample_T=1.0,
                            rng_seed=7,
                            )
kinodynamic_TI_unicycle_edge_bundle = gen_eb.generate_edge_bundle()
# eb_filename = 'edge_bundles/eb_unicycle_dbCBS_kinodynamic_TI_edges_100000.npz'
eb_filename = 'edge_bundles_unclamped/eb_unicycle_dbCBS_kinodynamic_TI_edges_100000.npz'
np.savez_compressed(eb_filename, **kinodynamic_TI_unicycle_edge_bundle)

gen_eb.plot_length_hist(None)
gen_eb.plot_edges_2d(None)



def get_start_state_kinematic_TI_unicycle(env, agent, rng):
    start_state = np.array([0.0, 0.0, 0.0])
    return start_state

from edge_bundle import GenerateEdgeBundle
gen_eb = GenerateEdgeBundle(env = env, 
                            agent=agent, 
                            num_edges=10000,
                            get_start_state=get_start_state_kinematic_TI_unicycle,
                            minimum_time_step=0.1,
                            max_sample_T=1.0,
                            rng_seed=77,
                            )
kinematic_TI_unicycle_edge_bundle = gen_eb.generate_edge_bundle()
# eb_filename = 'edge_bundles/eb_unicycle_dbCBS_kinematic_TI_edges_10000.npz'
eb_filename = 'edge_bundles_unclamped/eb_unicycle_dbCBS_kinematic_TI_edges_10000.npz'
np.savez_compressed(eb_filename, **kinematic_TI_unicycle_edge_bundle)

gen_eb.plot_length_hist(None)
gen_eb.plot_edges_2d(None)


"""


"""
#Second Order Car: Generate an edge bundle and save it 

import sys
sys.path.append('./src')
import numpy as np
import matplotlib.pyplot as plt
from Environments import SquareEnvironment, CircularObstacle2D
from Agents import SecondOrderCar
from edge_bundle import GenerateEdgeBundle
from mapf_env_square_agent_second_order_car import get_second_order_car_agent

obstacles = []
env = SquareEnvironment(40, 40, obstacles)
agent = get_second_order_car_agent(1)

def sample_with_extremes(rng, low, high, p_extreme=0.1):
    if rng.random() < p_extreme:
        # pick exactly low or high
        return low if rng.random() < 0.5 else high
    else:
        return rng.uniform(low, high)


def get_start_state_kinodynamic_TI_second_order_car(env, agent, rng):
    v   = sample_with_extremes(rng, -agent.max_speed, agent.max_speed, p_extreme=0.1)
    phi = sample_with_extremes(rng, -agent.max_phi,   agent.max_phi,   p_extreme=0.1)
    return np.array([0.0, 0.0, 0.0, v, phi], dtype=np.float64)

    
from edge_bundle import GenerateEdgeBundle
gen_eb = GenerateEdgeBundle(env = env, 
                            agent=agent, 
                            num_edges=100000,
                            get_start_state=get_start_state_kinodynamic_TI_second_order_car,
                            minimum_time_step=0.1,
                            max_sample_T=2.0,
                            rng_seed=77,
                            )
kinodynamic_TI_second_order_car_edge_bundle = gen_eb.generate_edge_bundle()
# eb_filename = 'edge_bundles/eb_second_order_car_kinodynamic_TI_edges_100000.npz'
eb_filename = 'edge_bundles_unclamped/eb_second_order_car_kinodynamic_TI_edges_100000.npz'
np.savez_compressed(eb_filename, **kinodynamic_TI_second_order_car_edge_bundle)

gen_eb.plot_length_hist(None)
gen_eb.plot_edges_2d(None)


from edge_bundle import GenerateEdgeBundle
gen_eb = GenerateEdgeBundle(env = env, 
                            agent=agent, 
                            num_edges=100000,
                            get_start_state=get_start_state_kinodynamic_TI_second_order_car,
                            minimum_time_step=0.1,
                            max_sample_T=1.0,
                            rng_seed=77,
                            )
kinodynamic_TI_second_order_car_edge_bundle = gen_eb.generate_edge_bundle()
eb_filename = 'edge_bundles/eb_second_order_car_kinodynamic_TI_edges_100000_t1.0.npz'
np.savez_compressed(eb_filename, **kinodynamic_TI_second_order_car_edge_bundle)

gen_eb.plot_length_hist(None)
gen_eb.plot_edges_2d(None)

"""



"""
#Quadcopter 6d: Generate an edge bundle and save it 

import sys
sys.path.append('./src')
import numpy as np
import matplotlib.pyplot as plt
from Environments import CuboidEnvironment
from Agents import QuadCopter6D
from edge_bundle import GenerateEdgeBundle
from mapf_env_cuboid_agent_quadcopter6d import get_quadcopter_agent

obstacles = []
env = CuboidEnvironment(40, 40, 40, obstacles)
agent = get_quadcopter_agent(1)


def sample_with_extremes(rng, low, high, p_extreme=0.1):
    if rng.random() < p_extreme:
        # pick exactly low or high
        return low if rng.random() < 0.5 else high
    else:
        return rng.uniform(low, high)


def get_start_state_kinodynamic_TI_quadcopter_6d(env, agent, rng):
    vx = sample_with_extremes(rng, -agent.max_speed, agent.max_speed, p_extreme=0.1)
    vy = sample_with_extremes(rng, -agent.max_speed, agent.max_speed, p_extreme=0.1)
    vz = sample_with_extremes(rng, -agent.max_speed, agent.max_speed, p_extreme=0.1)
    start_state = np.array([0.0, 0.0, 0.0, vx, vy, vz])
    return start_state

from edge_bundle import GenerateEdgeBundle
gen_eb = GenerateEdgeBundle(env = env, 
                            agent=agent, 
                            num_edges=200000,
                            get_start_state=get_start_state_kinodynamic_TI_quadcopter_6d,
                            minimum_time_step=0.1,
                            max_sample_T=1.0,
                            rng_seed=7,
                            )
kinodynamic_TI_quadcopter_6d_edge_bundle = gen_eb.generate_edge_bundle()
# eb_filename = 'edge_bundles/eb_quadcopter6d_kinodynamic_TI_edges_200000.npz'
eb_filename = 'edge_bundles_unclamped/eb_quadcopter6d_kinodynamic_TI_edges_200000.npz'
np.savez_compressed(eb_filename, **kinodynamic_TI_quadcopter_6d_edge_bundle)

gen_eb.plot_length_hist(None)
gen_eb.plot_edges_2d(None)


"""
