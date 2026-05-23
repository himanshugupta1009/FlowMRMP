import sys
sys.path.append('../src')
import numpy as np
import math

from Agents import UniCycle, QuadCopter6D, SecondOrderCar
from edge_bundle import EdgeBundle
from mapf_env_square_agent_unicycle import (
    ConstrainedUnicycleTrajOptOptions,
    load_unicycle_dbrrt_primitives,
    optimize_constrained_dbrrt_unicycle_path,
    transform_unicycle_trajectory_numba,
)
from mapf_env_cuboid_agent_quadcopter6d import (
    ConstrainedQuadcopter6DTrajOptOptions,
    load_quadcopter6d_motion_primitives,
    optimize_constrained_dbrrt_quadcopter6d_path,
    transform_quadcopter6d_trajectory_numba,
)
from kd_tree_unicycle import CircularAngleIndexNumba
from kd_tree_second_order_car import VPhiTree
from kd_tree_grid_quadcopter6d import VxyzGridTree


class AgentBuilder():
    """
    Abstract class for each agent builder

    All other agent builders must overload getters
    """
    def __init__(self):
        self.id = None
        self.seed = None
        self.edge_bundle = None
        self.kino_ti_edge_bundle = None
        self.dbcbs_name = None
        self.name = "Abstract"
        self.sampling_time_step = 1.0
        self.num_skip_edges = 10
        self.sort_edges = True

    def initialize_identification(self, agent_id, seed):
        """
        Sets the agent ID and seed for this agent 

        Args:
            agent_id (int): Unique agent ID
            seed (int): RNG seed 
        """
        self.id = agent_id
        self.seed = seed

    def get_agent(self):
        raise Exception("Do not use this class! Make a new agent builder that" \
        "inherits from this class for each specific agent ")
    
    def get_valid_function(self):
        raise Exception("Do not use this class! Make a new agent builder that" \
        "inherits from this class for each specific agent ")
    
    def get_valid_moving_obs_function(self):
        raise Exception("Do not use this class! Make a new agent builder that" \
        "inherits from this class for each specific agent ")
    
    def get_cost_function(self):
        raise Exception("Do not use this class! Make a new agent builder that" \
        "inherits from this class for each specific agent ")
    
    def get_random_pt_function(self):
        raise Exception("Do not use this class! Make a new agent builder that" \
        "inherits from this class for each specific agent ")
    
    def get_reached_goal_function(self):
        raise Exception("Do not use this class! Make a new agent builder that" \
        "inherits from this class for each specific agent ")
    
    def get_point_translate_function(self):
        raise Exception("Do not use this class! Make a new agent builder that" \
        "inherits from this class for each specific agent ")
    
    def get_edge_bundle(self):
        raise Exception("Do not use this class! Make a new agent builder that" \
        "inherits from this class for each specific agent ")
    
    def get_kino_ti_edge_bundle(self):
        raise Exception("No kino-ti edge bundle for this agent")
    
    def get_dbrrt_motion_primitives(self):
        raise NotImplementedError("No dbRRT motion primitives for this agent")

    def get_dbrrt_transform_function(self):
        raise NotImplementedError(
            "No dbRRT trajectory transform function for this agent")

    def get_dbrrt_optimizer_function(self):
        raise NotImplementedError("No dbRRT optimizer function for this agent")

    def get_sort_edges_func(self):
        raise Exception("Do not use this class! Make a new agent builder that" \
        "inherits from this class for each specific agent ")
    
    def get_start(self, env_length, env_bredth, buffer, loc_rng, x=None, y=None, t=None):
        raise Exception("Do not use this class! Make a new agent builder that" \
        "inherits from this class for each specific agent ")
    
    def get_agent_declaration(self):
        raise Exception("Do not use this class! Make a new agent builder that" \
        "inherits from this class for each specific agent ")

    def get_manifest_config(self):
        excluded = {"id", "seed", "edge_bundle", "kino_ti_edge_bundle"}
        params = {}
        for key, value in sorted(self.__dict__.items()):
            if key in excluded:
                continue
            if key.startswith("_"):
                continue
            params[key] = value

        return {
            "class": self.__class__.__name__,
            "name": self.name,
            "params": params,
        }


class UnicycleBuilder(AgentBuilder):
    """
    AgentBuilder for the Unicycle agent
    """
    edge_bundles = {}
    kino_ti_edge_bundles = {}
    dbrrt_motion_primitives = {}

    def __init__(self, max_speed=0.5, max_omega=0.5, radius=0.3,
                 edge_bundle_file_location = 'edge_bundles/eb_unicycle_dbCBS_kinematic_TI_edges_10000.npz',
                 kino_ti_edge_bundle_file_location = 'edge_bundles_unclamped/eb_unicycle_dbCBS_kinodynamic_TI_edges_100000.npz',
                 sampling_time_step=1.0,
                 num_edges = 1000, #From the kinematic edge bundle
                 kd_num_edges = 30000, #From the kinodynamic edge bundle
                 num_skip_edges=10, #Used for both kinematic and kinodynamic edge bundles
                 motion_primitive_file_location = 'motion_primitives/unicycle1_v0__ispso__2023_04_03__14_56_57.bin.im.bin.im.bin.msgpack',
                 num_motion_primitives=30000, #From the dbRRT motion primitives
                 motion_primitive_dt=0.1, #dbRRT motion primitive time step
                 ):
        super().__init__()

        self.dbcbs_name = 'unicycle1_sphere_v0'
        self.name = "UCYCLE"
        self.max_speed = max_speed
        self.max_omega = max_omega
        self.radius = radius
        self.sampling_time_step = sampling_time_step
        self.num_skip_edges = num_skip_edges

        self.num_edges = num_edges
        self.edge_bundle_file_location = edge_bundle_file_location

        self.kd_num_edges = kd_num_edges
        self.kino_ti_edge_bundle_file_location = kino_ti_edge_bundle_file_location

        self.num_motion_primitives = num_motion_primitives
        self.motion_primitive_dt = motion_primitive_dt
        self.motion_primitive_file_location = motion_primitive_file_location


    def get_agent(self):
        """
        Gets a new UniCycle agent 

        Returns:
            UniCycle: new agent with assigned rng, seed
        """
        return UniCycle(max_speed = self.max_speed,
                 max_omega= self.max_omega,
                 agent_id=self.id,
                 rng_seed=self.seed,
                 radius=self.radius)
    
    def get_valid_function(self):
        return UniCycle.is_new_node_valid
    
    def get_valid_moving_obs_function(self):
        return UniCycle.is_new_node_valid
    
    def get_cost_function(self):
        return UniCycle.get_cost
    
    def get_random_pt_function(self):
        return UniCycle.get_random_point
    
    def get_reached_goal_function(self):
        return UniCycle.agent_reached_goal
    
    def get_point_translate_function(self):
        return UniCycle.point_translate_function
    
    def get_sort_edges_func(self):
        return UniCycle.sort_edges

    def get_edge_bundle(self):
        if self.seed in UnicycleBuilder.edge_bundles:
            return UnicycleBuilder.edge_bundles[self.seed]

        data = np.load(self.edge_bundle_file_location, allow_pickle=True)
        eb = EdgeBundle(data, fix_num_edges=self.num_edges,
                        rng_seed=self.seed * 67, use_all_edges=False)
        UnicycleBuilder.edge_bundles[self.seed] = eb
        return eb
    
    def get_kino_ti_edge_bundle(self):
        if self.seed in UnicycleBuilder.kino_ti_edge_bundles:
            return UnicycleBuilder.kino_ti_edge_bundles[self.seed]

        data = np.load(self.kino_ti_edge_bundle_file_location, allow_pickle=True)
        kino_ti_edge_bundle = EdgeBundle(data, fix_num_edges=self.kd_num_edges,
                                    rng_seed=self.seed * 67, use_all_edges=False)
        edge_ids = np.arange(kino_ti_edge_bundle.num_edges, dtype=np.int64)
        thetas = kino_ti_edge_bundle.start_states[:, 2]  # heading angle θ
        kd_tree_ti_edge_bundle = CircularAngleIndexNumba(thetas, ids=edge_ids)

        eb = (kino_ti_edge_bundle, kd_tree_ti_edge_bundle)
        UnicycleBuilder.kino_ti_edge_bundles[self.seed] = eb
        return eb
    
    def get_dbrrt_motion_primitives(self):
        cache_key = (
            self.motion_primitive_file_location,
            self.num_motion_primitives,
            self.motion_primitive_dt,
        )
        if cache_key not in UnicycleBuilder.dbrrt_motion_primitives:
            UnicycleBuilder.dbrrt_motion_primitives[cache_key] = (
                load_unicycle_dbrrt_primitives(
                    num_edges=self.num_motion_primitives,
                    dt=self.motion_primitive_dt,
                    primitive_file_location=self.motion_primitive_file_location,
                )
            )
        return UnicycleBuilder.dbrrt_motion_primitives[cache_key]

    def get_dbrrt_transform_function(self):
        return transform_unicycle_trajectory_numba

    def get_dbrrt_optimizer_function(self):
        return lambda curr_planner: optimize_constrained_dbrrt_unicycle_path(
            curr_planner,
            options=ConstrainedUnicycleTrajOptOptions(allow_raw_fallback=False),
        )

    def get_start(self, env_width, env_bredth, buffer,
                  loc_rng, x=None, y=None, t=None):
        if x is None:
            x = loc_rng.uniform(buffer, env_width-buffer) 
        if y is None:
            y = loc_rng.uniform(buffer, env_bredth-buffer)
        if t is None:
            t = loc_rng.uniform(0, 2 * np.pi) 
        return (x, y, t)
    
    def get_agent_declaration(self):
        return f"UniCycle(max_speed={self.max_speed}, agent_id={self.id}, rng_seed={self.seed},max_omega={self.max_omega}, radius={self.radius})"
    
    
class SecondOrderCarBuilder(AgentBuilder):
    """
    AgentBuilder for the SOC agent
    """
    kino_ti_edge_bundles = {}

    def __init__(self, max_speed=1., max_acceleration = 2., 
                 max_phi = np.pi/3, max_steering_rate = 0.5,
                 wheelbase=0.7, radius=0.3, 
                 kino_ti_edge_bundle_file_location='edge_bundles_unclamped/eb_second_order_car_kinodynamic_TI_edges_100000.npz', 
                 sampling_time_step=2.0,
                 kd_num_edges=50000, #From the kinodynamic edge bundle
                 num_skip_edges=10, #Used for kinodynamic edge bundles
                 sort_edges=True
                 ):
        super().__init__()
        
        self.max_speed = max_speed
        self.radius = radius
        self.max_acceleration = max_acceleration
        self.max_phi = max_phi
        self.max_steering_rate = max_steering_rate
        self.wheelbase = wheelbase
        self.sampling_time_step = sampling_time_step
        self.num_skip_edges = num_skip_edges

        self.name = "SOC"

        self.kino_ti_edge_bundle_file_location = kino_ti_edge_bundle_file_location
        self.kd_num_edges = kd_num_edges
        self.sort_edges = sort_edges

    def get_agent(self):
        return SecondOrderCar(radius=self.radius, 
                              agent_id=self.id, 
                              rng_seed=self.seed, 
                              max_phi=self.max_phi, 
                              max_speed=self.max_speed, 
                              max_acceleration=self.max_acceleration,
                              max_steering_rate=self.max_steering_rate,
                              wheelbase=self.wheelbase)
    
    def get_valid_function(self):
        return SecondOrderCar.is_new_node_valid
    
    def get_valid_moving_obs_function(self):
        return SecondOrderCar.is_new_node_valid
    
    def get_cost_function(self):
        return SecondOrderCar.get_cost
    
    def get_random_pt_function(self):
        return SecondOrderCar.get_random_point
    
    def get_reached_goal_function(self):
        return SecondOrderCar.agent_reached_goal
    
    def get_point_translate_function(self):
        return SecondOrderCar.kd_tree_point_translate_function
    
    def get_edge_bundle(self):
        raise Exception("Non-Kino Edge Bundles don't work with the Second Order Car")
    
    def get_kino_ti_edge_bundle(self):
        if self.seed in SecondOrderCarBuilder.kino_ti_edge_bundles:
            return SecondOrderCarBuilder.kino_ti_edge_bundles[self.seed]

        data = np.load(self.kino_ti_edge_bundle_file_location, allow_pickle=True)
        kino_ti_edge_bundle = EdgeBundle(data, fix_num_edges=self.kd_num_edges,
                                    use_all_edges=False, rng_seed=67 * self.seed)
        edge_ids = np.arange(kino_ti_edge_bundle.num_edges, dtype=np.int64)
        speeds = kino_ti_edge_bundle.start_states[:, 3]  # v
        phis = kino_ti_edge_bundle.start_states[:, 4]   # phi
        v_scale = self.max_speed
        phi_scale = self.max_phi
        kd_tree_ti_edge_bundle = VPhiTree(speeds, phis, ids=edge_ids, 
                    v_scale=v_scale, phi_scale=phi_scale)
        
        eb = (kino_ti_edge_bundle, kd_tree_ti_edge_bundle)
        SecondOrderCarBuilder.kino_ti_edge_bundles[self.seed] = eb
        return eb
    
    def get_start(self, env_width, env_bredth, buffer,
                  loc_rng, x=None, y=None, t=None):
        if x is None:
            x = loc_rng.uniform(buffer, env_width-buffer) 
        if y is None:
            y = loc_rng.uniform(buffer, env_bredth-buffer)
        if t is None:
            t = loc_rng.uniform(0, 2 * np.pi) 
        return (x, y, t, 0., 0.)
    
    def get_agent_declaration(self):
        return f"SecondOrderCar(max_speed={self.max_speed}, agent_id={self.id}, rng_seed={self.seed}, max_phi={self.max_phi}, radius={self.radius}, max_steering_rate={self.max_steering_rate}, wheelbase={self.wheelbase})"

    
class QuadcopterBuilder(AgentBuilder):
    """
    AgentBuilder for the quadcopter agent
    """
    kino_ti_edge_bundles = {}
    dbrrt_motion_primitives = {}

    def __init__(self, max_speed=0.5, max_acceleration = 2., radius=0.1,
                 kino_ti_edge_bundle_file_location='edge_bundles_unclamped/eb_quadcopter6d_kinodynamic_TI_edges_200000.npz', 
                 sampling_time_step=1.0,
                 kd_num_edges=100000, #From the kinodynamic edge bundle
                 num_skip_edges=10, #Used for kinodynamic edge bundles
                 motion_primitive_file_location='motion_primitives/quadcopter6d_long_50_1000_primitives.npz',
                 num_motion_primitives=1000, #From the dbRRT motion primitives
                 motion_primitive_dt=0.1, #dbRRT motion primitive time step
                ):
        super().__init__()
        
        self.max_speed = max_speed
        self.radius = radius
        self.max_acceleration = max_acceleration
        self.num_skip_edges = num_skip_edges
        self.sampling_time_step = sampling_time_step

        self.name = "QUAD"
        self.dbcbs_name = 'integrator2_3d_v0'

        self.kino_ti_edge_bundle_file_location = kino_ti_edge_bundle_file_location
        self.kd_num_edges = kd_num_edges

        self.num_motion_primitives = num_motion_primitives
        self.motion_primitive_dt = motion_primitive_dt
        self.motion_primitive_file_location = motion_primitive_file_location

    def get_agent(self):
        return QuadCopter6D(radius=self.radius, 
                              agent_id=self.id, 
                              rng_seed=self.seed, 
                              max_speed=self.max_speed, 
                              max_acceleration=self.max_acceleration)
    
    def get_valid_function(self):
        return QuadCopter6D.is_new_node_valid
    
    def get_valid_moving_obs_function(self):
        return QuadCopter6D.is_new_node_valid
    
    def get_cost_function(self):
        return QuadCopter6D.get_cost
    
    def get_random_pt_function(self):
        return QuadCopter6D.get_random_point
    
    def get_reached_goal_function(self):
        return QuadCopter6D.agent_reached_goal
    
    def get_point_translate_function(self):
        return QuadCopter6D.kd_tree_point_translate_function
    
    def get_edge_bundle(self):
        raise Exception("Non-Kino Edge Bundles don't work with the Second Order Car")
    
    def get_kino_ti_edge_bundle(self):
        if self.seed in QuadcopterBuilder.kino_ti_edge_bundles:
            return QuadcopterBuilder.kino_ti_edge_bundles[self.seed]

        data = np.load(self.kino_ti_edge_bundle_file_location, allow_pickle=True)
        kino_ti_edge_bundle = EdgeBundle(data, fix_num_edges=self.kd_num_edges,
                                    use_all_edges=False, rng_seed=67 * self.seed)
        edge_ids = np.arange(kino_ti_edge_bundle.num_edges, dtype=np.int64)
        vx = kino_ti_edge_bundle.start_states[:, 3]  # vx
        vy = kino_ti_edge_bundle.start_states[:, 4]  # vy
        vz = kino_ti_edge_bundle.start_states[:, 5]  # vz
        v_scale = 1.0
        delta_radius = 0.1
        kd_tree_TI_eb = VxyzGridTree(vx, vy, vz, ids=edge_ids, 
                        scales=(v_scale,v_scale,v_scale),
                        vmin=-self.max_speed,
                        vmax= self.max_speed,
                        cell_size=delta_radius/2,
                        initial_out_capacity=2048,
                        return_ids=False
                        )
        
        eb = (kino_ti_edge_bundle, kd_tree_TI_eb)
        QuadcopterBuilder.kino_ti_edge_bundles[self.seed] = eb
        return eb

    def get_dbrrt_motion_primitives(self):
        cache_key = (
            self.motion_primitive_file_location,
            self.num_motion_primitives,
            self.motion_primitive_dt,
        )
        if cache_key not in QuadcopterBuilder.dbrrt_motion_primitives:
            QuadcopterBuilder.dbrrt_motion_primitives[cache_key] = (
                load_quadcopter6d_motion_primitives(
                    num_edges=self.num_motion_primitives,
                    dt=self.motion_primitive_dt,
                    primitive_file_location=self.motion_primitive_file_location,
                )
            )
        return QuadcopterBuilder.dbrrt_motion_primitives[cache_key]

    def get_dbrrt_transform_function(self):
        return transform_quadcopter6d_trajectory_numba

    def get_dbrrt_optimizer_function(self):
        return lambda curr_planner: optimize_constrained_dbrrt_quadcopter6d_path(
            curr_planner,
            options=ConstrainedQuadcopter6DTrajOptOptions(),
        )

    def get_start(self, env_width, env_bredth, env_depth,
                  buffer, loc_rng, x=None, y=None, z=None):
        if x is None:
            x = loc_rng.uniform(buffer, env_width-buffer) 
        if y is None:
            y = loc_rng.uniform(buffer, env_bredth-buffer)
        if z is None:
            z = loc_rng.uniform(buffer, env_depth-buffer) 
        return (x, y, z, 0., 0., 0.)
    
    def get_agent_declaration(self):
        return f"QuadCopter6D(max_speed={self.max_speed}, agent_id={self.id}, rng_seed={self.seed}, radius={self.radius}, max_acceleration={self.max_acceleration})"
