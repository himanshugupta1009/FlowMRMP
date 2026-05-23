import contextlib
import os
import sys
import time

import numpy as np
import yaml

sys.path.append('src')

from Environments import *
from cRRT import CRRT
from cRRT_eb import CRRT_EBType2
from constrainedX import (
    ConstrainedDbRRTPlanner,
    ConstrainedEdgeBundleType2RRT,
    ConstrainedKinoTIEBRRT,
    ConstrainedRRT,
)
from kcbs import KCBS, check_high_resolution_paths_collision_free
from kinodynamic_TI_eb_crrt import KinoTIEBCRRT
from kinodynamic_TI_eb_rrt import *
from printer import MultiRRTPrinter
from prioritized_planning import PrioritizedPlanning
from prrt_eb import EdgeBundlePRRT
from rrt import RRT


def _raise_if_highres_paths_collide(
        highres_paths,
        agent_objs,
        distance_metric_state_size,
        dynamic_agent_clearance=0.0,
        roundoff_digits=1):
    for agent in agent_objs:
        if agent.distance_metric_state_size != distance_metric_state_size:
            raise ValueError(
                "All agents must have the same distance_metric_state_size "
                "for collision checking.")

    collision_result = check_high_resolution_paths_collision_free(
        highres_paths,
        agents=agent_objs,
        distance_metric_state_size=distance_metric_state_size,
        dynamic_agent_clearance=dynamic_agent_clearance,
        roundoff_digits=roundoff_digits,
    )
    if collision_result["collision_free"]:
        return

    collision_times = collision_result["collision_times"]
    if len(collision_times) > 0:
        collision_range = str(collision_times[0]) + " - " + str(collision_times[-1])
    else:
        collision_range = "unknown"
    raise Exception(
        "Conflict for range " + collision_range +
        " for agents " + str(collision_result["first_agent"]) +
        " and " + str(collision_result["second_agent"]))


def _common_rrt_runtime_params(test_class):
    return {
        "max_iter": np.inf,
        "planning_time": test_class.max_planning_time,
        "udf_seed": "round seed",
        "sampling_time_step": _runtime_param(
            test_class, "min_sampling_time_step",
            "min(agent.sampling_time_step)"),
        "use_fixed_sampling_time": False,
        "minimum_time_step": 0.1,
    }


def _common_kcbs_params(test_class):
    return {
        "max_trials": np.inf,
        "planning_time": test_class.max_planning_time,
        "clearance_threshold": test_class.dynamic_agent_clearance,
        "rng_seed": "round seed",
    }


def _runtime_param(test_class, key, source):
    runtime_values = getattr(test_class, "_manifest_runtime_values", {})
    value = {"source": source}
    if key in runtime_values:
        value["value"] = runtime_values[key]
    return value


def _manifest_planner_constructor_params(test_class):
    class_name = test_class.__class__.__name__

    if class_name == "CRRTTestClass":
        params = _common_rrt_runtime_params(test_class)
        params.update({
            "planner_class": "CRRT",
            "truncate_paths": test_class.truncate_paths,
            "branch_goal_parking": test_class.branch_goal_parking,
            "truncation_check_threshold": test_class.truncation_check_threshold,
            "num_extension_trials": test_class.num_extension_trials,
            "dynamic_agent_clearance": test_class.dynamic_agent_clearance,
        })
        return params

    if class_name == "CRRTEBTestClass":
        params = _common_rrt_runtime_params(test_class)
        params.update({
            "planner_class": "CRRT_EBType2",
            "num_skip_edges": _runtime_param(
                test_class, "min_num_skip_edges",
                "min(agent.num_skip_edges)"),
            "num_random_edges": 1,
            "dynamic_agent_clearance": test_class.dynamic_agent_clearance,
        })
        return params

    if class_name == "KinoTiCRRTEBTestClass":
        params = _common_rrt_runtime_params(test_class)
        params.update({
            "planner_class": "KinoTIEBCRRT",
            "max_num_edges_per_node": test_class.max_num_edges_per_node,
            "num_extension_trials": test_class.num_extension_trials,
            "num_edge_candidates_per_agent": test_class.num_edge_candidates_per_agent,
            "max_joint_edge_trials": test_class.max_joint_edge_trials,
            "epsilon_random": test_class.epsilon_random,
            "fallback_to_random_control": test_class.fallback_to_random_control,
            "kd_tree_delta_radius": test_class.kd_delta_radius,
            "dynamic_agent_clearance": test_class.dynamic_agent_clearance,
            "truncate_paths": test_class.truncate_paths,
            "branch_goal_parking": test_class.branch_goal_parking,
        })
        return params

    if class_name == "PrrtTestClass":
        return {
            "planner_class": "RRT",
            "wrapper_class": "PrioritizedPlanning",
            "per_agent_planner": {
                "use_fixed_sampling_time": False,
                "sampling_time_step": _runtime_param(
                    test_class, "min_sampling_time_step",
                    "min(agent.sampling_time_step)"),
                "minimum_time_step": 0.1,
                "max_iter": np.inf,
                "planning_time": test_class.max_planning_time,
                "num_extension_trials": test_class.num_extension_trials,
                "udf_seed": "seed_rng.integers(...) per agent",
                "goal_sampling_probability": test_class.goal_sampling_probability,
                "dynamic_agent_clearance": test_class.dynamic_agent_clearance,
            },
            "prioritized_planning": {
                "planning_time": test_class.max_planning_time,
            },
        }

    if class_name == "PrrtEbTestClass":
        return {
            "planner_class": "EdgeBundlePRRT",
            "max_iter": np.inf,
            "planning_time": test_class.max_planning_time,
            "num_skip_edges": _runtime_param(
                test_class, "min_num_skip_edges",
                "min(agent.num_skip_edges)"),
            "num_rand_edges": 1,
            "udf_seed": "round seed",
            "use_fixed_sampling_time": False,
            "sampling_time_step": _runtime_param(
                test_class, "min_sampling_time_step",
                "min(agent.sampling_time_step)"),
            "dynamic_agent_clearance": test_class.dynamic_agent_clearance,
        }

    if class_name == "PrioritizedKinoTIRRTTestClass":
        return {
            "planner_class": "KinoTIEBRRT",
            "wrapper_class": "PrioritizedPlanning",
            "per_agent_planner": {
                "use_fixed_sampling_time": False,
                "sampling_time_step": _runtime_param(
                    test_class, "agent_sampling_time_step",
                    "agent.sampling_time_step"),
                "minimum_time_step": 0.1,
                "max_iter": np.inf,
                "planning_time": test_class.max_planning_time,
                "max_num_edges_per_node": test_class.max_num_edges_per_node,
                "num_skip_edges": _runtime_param(
                    test_class, "agent_num_skip_edges",
                    "agent.num_skip_edges"),
                "num_random_edges": test_class.num_extension_trials,
                "kd_tree_delta_radius": test_class.kd_delta_radius,
                "udf_seed": "agent.seed * round seed",
                "goal_sampling_probability": test_class.goal_sampling_probability,
                "dynamic_agent_clearance": test_class.dynamic_agent_clearance,
                "epsilon_random": test_class.epsilon_random,
            },
            "prioritized_planning": {
                "planning_time": test_class.max_planning_time,
            },
        }

    if class_name == "KcbsTestClass":
        return {
            "high_level_planner": {
                "planner_class": "KCBS",
                **_common_kcbs_params(test_class),
            },
            "low_level_planner": {
                "planner_class": "ConstrainedRRT",
                "sampling_time_step": _runtime_param(
                    test_class, "agent_sampling_time_step",
                    "agent.sampling_time_step"),
                "minimum_time_step": 0.1,
                "max_iter": test_class.num_low_level_planner_iterations,
                "num_extension_trials": test_class.num_extension_trials,
                "planning_time": 2 * test_class.max_planning_time,
                "udf_seed": "agent.seed * round seed",
                "use_fixed_sampling_time": False,
            },
        }

    if class_name == "KcbsEbTestClass":
        return {
            "high_level_planner": {
                "planner_class": "KCBS",
                **_common_kcbs_params(test_class),
            },
            "low_level_planner": {
                "planner_class": "ConstrainedEdgeBundleType2RRT",
                "sampling_time_step": _runtime_param(
                    test_class, "agent_sampling_time_step",
                    "agent.sampling_time_step"),
                "minimum_time_step": 0.1,
                "max_iter": 10000,
                "num_random_edges": 1,
                "num_skip_edges": _runtime_param(
                    test_class, "agent_num_skip_edges",
                    "agent.num_skip_edges"),
                "planning_time": 2 * test_class.max_planning_time,
                "udf_seed": "agent.seed * round seed",
                "use_fixed_sampling_time": False,
            },
        }

    if class_name == "KcbsKinoTiEbTestClass":
        return {
            "high_level_planner": {
                "planner_class": "KCBS",
                **_common_kcbs_params(test_class),
            },
            "low_level_planner": {
                "planner_class": "ConstrainedKinoTIEBRRT",
                "use_fixed_sampling_time": False,
                "sampling_time_step": _runtime_param(
                    test_class, "agent_sampling_time_step",
                    "agent.sampling_time_step"),
                "minimum_time_step": 0.1,
                "max_iter": test_class.num_low_level_planner_iterations,
                "planning_time": 2 * test_class.max_planning_time,
                "max_num_edges_per_node": test_class.max_num_edges_per_node,
                "num_skip_edges": _runtime_param(
                    test_class, "agent_num_skip_edges",
                    "agent.num_skip_edges"),
                "epsilon_random": test_class.epsilon_random,
                "num_random_edges": test_class.num_extension_trials,
                "kd_tree_delta_radius": test_class.kd_delta_radius,
                "udf_seed": "overwritten by KCBS init",
            },
        }

    if class_name == "KcbsDbrrtTestClass":
        return {
            "high_level_planner": {
                "planner_class": "KCBS",
                **_common_kcbs_params(test_class),
            },
            "low_level_planner": {
                "planner_class": "ConstrainedDbRRTPlanner",
                "alpha": test_class.dbrrt_alpha,
                "delta": test_class.dbrrt_delta,
                "minimum_time_step": 0.1,
                "max_iter": test_class.num_low_level_planner_iterations,
                "planning_time": 2 * test_class.max_planning_time,
                "max_candidate_motions_per_expand": test_class.max_candidate_motions_per_expand,
                "allow_intermediate_goal": True,
                "cost_delta_factor": test_class.cost_delta_factor,
                "goal_bias": test_class.goal_sampling_probability,
                "goal_expand_mode": "focused",
                "random_expand_mode": "randomized",
                "dynamic_agent_clearance": test_class.dynamic_agent_clearance,
                "udf_seed": "overwritten by KCBS init",
                "use_optimizer": test_class.use_optimizer,
            },
        }

    return {}


class AbstractTestClass:
    def __init__(self, printenv=False, max_planning_time=300.,
                 print_logs = False, debug_flag = False, 
                 obs_buffers = True, savepath = ""):
        """
        Abstract test class--all test classes must inherit from this and implement test_func
        Also holds common data structures to store results
        
        Args:
            printenv (bool): whether to print environment visualizations
            max_planning_time (float): maximum planning time for each test
            print_logs (bool): whether to print logs during planning
            debug_flag (bool): whether to enable debug mode during planning
            obs_buffers (bool): whether to use obstacle buffers in the environment
            savepath (str): path to save environment visualizations
        """

        # metrics for each test class, accessed by pipeline 
        self.name = "Don't Instantiate this" # name set in test class constructors
        self.short_name = "Don't Instantiate this" # short name set in test class constructors  
        self.exceptions = {}
        self.costs = {}
        self.times = {}
        self.path_times = {}
        self.max_times = {}
        self.success = {}
        self.messages = {}
        self.printenv = printenv
        self.max_planning_time = max_planning_time
        self.print_logs = print_logs
        self.debug_flag = debug_flag
        self.obs_buffers = obs_buffers
        self.savepath = ("" if savepath is None else savepath) + ("" if savepath == "" else "/") + "pics/"
        if printenv:
            os.makedirs(self.savepath, exist_ok=True)

    # method sig for test class runner--each must match this *exactly*
    def test_func(self, agents, starts, obstacles, goals, goal_radii,
                  seed, env_width, env_depth,  env_height):
        raise Exception("Don't instantiate this")

    def warmup_test(self, agents, starts, obstacles, goals, goal_radii, 
                    seed, env_width, env_depth, env_height=None):
        """
        Warm up function to pre-compile numba functions
        """
        self.max_planning_time, old_time = 5, self.max_planning_time
        with open('/dev/null', 'w') as f, contextlib.redirect_stdout(f):
            self.test_func(agents, starts, obstacles, goals, goal_radii,
                           seed, env_width, env_depth, env_height=env_height)
        self.max_planning_time = old_time

    def get_manifest_config(self, runtime_values=None):
        excluded = {
            "exceptions",
            "costs",
            "times",
            "path_times",
            "max_times",
            "success",
            "messages",
            "printenv",
            "print_logs",
            "debug_flag",
            "savepath",
        }
        params = {}
        for key, value in sorted(self.__dict__.items()):
            if key in excluded:
                continue
            if key.startswith("_"):
                continue
            if key in {"name", "short_name"}:
                continue
            params[key] = value

        old_runtime_values = getattr(self, "_manifest_runtime_values", None)
        self._manifest_runtime_values = runtime_values or {}
        try:
            return {
                "class": self.__class__.__name__,
                "name": self.name,
                "short_name": self.short_name,
                "params": params,
                "planner_constructor_params": _manifest_planner_constructor_params(self),
            }
        finally:
            if old_runtime_values is None:
                delattr(self, "_manifest_runtime_values")
            else:
                self._manifest_runtime_values = old_runtime_values
        

class CRRTTestClass(AbstractTestClass):
    """
    Test class for CRRT
    """
    def __init__(self, printenv = False, collision_checks = False,
                max_planning_time=300.0, print_logs = False,
                debug_flag = False, obs_buffers = True,
                truncate_paths = False, branch_goal_parking = True,
                num_extension_trials=10,
                truncation_check_threshold=1.0,
                dynamic_agent_clearance=0.0):
        if branch_goal_parking and truncate_paths:
            raise ValueError(
                "branch_goal_parking requires truncate_paths=False")
        super().__init__(printenv, max_planning_time=max_planning_time,
                         print_logs=print_logs, debug_flag=debug_flag,
                         obs_buffers=obs_buffers)
        self.name = "CRRT"
        self.short_name = "CRRT"
        self.collision_checks = collision_checks
        self.truncate_paths = truncate_paths
        self.branch_goal_parking = branch_goal_parking
        self.num_extension_trials = num_extension_trials
        self.truncation_check_threshold = truncation_check_threshold
        self.dynamic_agent_clearance = dynamic_agent_clearance

    """
    Each test class must have a similar 'test_func' that takes the same arguments 
    """
    def test_func(self, agents, starts, obstacles, goals, goal_radii,
                  seed, env_width, env_depth, env_height=None):
        print("Testing " + self.name)

        if env_height is None:
            env = SquareEnvironment(env_width, env_depth, obstacles,
                                    obs_buffers=self.obs_buffers)
        else:
            env = CuboidEnvironment(env_width, env_depth, env_height,
                            obstacles, obs_buffers=self.obs_buffers)
        
        agent_objs = []
        isvalid_funcs = []
        cost_funcs = []
        random_pt_funcs = [] 
        reached_goal_funcs = []
        sampling_time_steps = [] 
        within_max_time = True
        success = True

        for agent in agents:
            agent_objs.append(agent.get_agent())
            isvalid_funcs.append(agent.get_valid_function())
            cost_funcs.append(agent.get_cost_function())
            random_pt_funcs.append(agent.get_random_pt_function())
            reached_goal_funcs.append(agent.get_reached_goal_function())
            sampling_time_steps.append(agent.sampling_time_step)

        # instantiate CRRT obj
        crrt = CRRT(agents=agent_objs, 
                    starts=starts,
                    goals=goals,
                    goal_radii=goal_radii,
                    env=env,
                    max_iter = np.inf, 
                    planning_time=self.max_planning_time,          
                    isvalid_function=isvalid_funcs, 
                    cost_function=cost_funcs,
                    random_point_function=random_pt_funcs, 
                    reached_goal_function = reached_goal_funcs,
                    udf_seed = seed, 
                    truncate_paths=self.truncate_paths,
                    branch_goal_parking=self.branch_goal_parking,
                    truncation_check_threshold=self.truncation_check_threshold,
                    num_extension_trials=self.num_extension_trials,
                    dynamic_agent_clearance=self.dynamic_agent_clearance,
                    print_logs=self.print_logs,
                    debug_flag=self.debug_flag, 
                    sampling_time_step=min(sampling_time_steps),
                    use_fixed_sampling_time=False,
                    minimum_time_step=0.1,
                    )
        # plan path 
        time = crrt.plan_path()

        if time > self.max_planning_time:
            within_max_time = False
            print("CRRT time overflow: found time of ", time)
        time = min(time, self.max_planning_time)

        if (not crrt.path_found) or (not within_max_time):
            success=False

        highres_paths = dict(enumerate(crrt.get_high_resolution_paths()))
        paths, states, controls, timesteps, costs = crrt.get_path()

        if(self.printenv):
            pcol = ['xkcd:powder pink', 'xkcd:metallic blue', 'xkcd:pastel orange',
                    'xkcd:pastel blue', 'xkcd:terracotta', 'xkcd:purplish grey',
                    'xkcd:powder pink', 'xkcd:metallic blue', 'xkcd:pastel orange',
                    'xkcd:pastel blue', 'xkcd:terracotta', 'xkcd:purplish grey']
            if success:
                mprint = MultiRRTPrinter(env, crrt, paths, pcol, pcol, joint_states=True)
                mprint.print_rrt('crrt_pipeline_' + str(seed) + "_" + str(len(agents)) + '.png',
                                 print_tree=False)
                mprint.print_highres_simulation(highres_paths,
                'crrt_pipeline_' + str(seed) + "_" + str(len(agents)) + '.gif', animation_speed=100)


        if not success:
            return (False, time, 0, 0, 0, "") 

        if self.collision_checks:
            _raise_if_highres_paths_collide(
                crrt.get_high_resolution_path_numpy_array(),
                agent_objs,
                agent_objs[0].distance_metric_state_size,
                crrt.dynamic_agent_clearance,
                crrt.roundoff_digits)
            
        max_time = crrt.path_time
        costs = crrt.path_cost

        return (True, time, sum(costs), crrt.path_time, max_time, "")
    

class CRRTEBTestClass(AbstractTestClass):
    """
    Test class for CRRT with Type2 EB
    """
    def __init__(self, printenv = False, collision_checks = False,
                 max_planning_time=300.0, print_logs = False,
                 debug_flag = False, obs_buffers = True,
                 dynamic_agent_clearance=0.0):
        
        super().__init__(printenv, max_planning_time=max_planning_time,
                         print_logs=print_logs, debug_flag=debug_flag,
                         obs_buffers=obs_buffers)
        self.name = "CRRT with Type 2 Edge Bundles"
        self.short_name = "EB CRRT"
        self.collision_checks = collision_checks
        self.dynamic_agent_clearance = dynamic_agent_clearance

    """
    Each test class must have a similar 'test_func' that takes the same arguments 
    """
    def test_func(self, agents, starts, obstacles, goals, goal_radii,
                  seed, env_width, env_depth, env_height=None):
        print("Testing " + self.name)

        if env_height is None:
            env = SquareEnvironment(env_width, env_depth, obstacles,
                                    obs_buffers=self.obs_buffers)
        else:
            env = CuboidEnvironment(env_width, env_depth, env_height,
                            obstacles, obs_buffers=self.obs_buffers)
        
        agent_objs = []
        ebs = []
        isvalid_funcs = []
        cost_funcs = []
        random_pt_funcs = [] 
        reached_goal_funcs = [] 
        translate_funcs = [] 
        sort_edges_funcs = []
        sampling_time_steps = []
        nums_skip_edges = []
        within_max_time = True
        success=True

        for agent in agents:
            agent_objs.append(agent.get_agent())
            ebs.append(agent.get_edge_bundle())
            isvalid_funcs.append(agent.get_valid_function())
            cost_funcs.append(agent.get_cost_function())
            random_pt_funcs.append(agent.get_random_pt_function())
            reached_goal_funcs.append(agent.get_reached_goal_function())
            translate_funcs.append(agent.get_point_translate_function())
            sort_edges_funcs.append(agent.get_sort_edges_func())
            sampling_time_steps.append(agent.sampling_time_step)
            nums_skip_edges.append(agent.num_skip_edges)

        # instantiate CRRT obj
        crrt = CRRT_EBType2(agents=agent_objs, 
                    starts=starts,
                    goals=goals,
                    goal_radii=goal_radii,
                    env=env,
                    edge_bundle=ebs,
                    max_iter = np.inf, 
                    planning_time=self.max_planning_time,          
                    isvalid_function=isvalid_funcs, 
                    cost_function=cost_funcs,
                    random_point_function=random_pt_funcs, 
                    reached_goal_function = reached_goal_funcs,
                    translate_function=translate_funcs,
                    sort_edges_function=sort_edges_funcs,
                    num_skip_edges=min(nums_skip_edges),
                    num_random_edges=1,
                    udf_seed = seed, 
                    dynamic_agent_clearance=self.dynamic_agent_clearance,
                    print_logs=self.print_logs,
                    debug_flag=self.debug_flag, 
                    sampling_time_step=min(sampling_time_steps),
                    use_fixed_sampling_time=False,
                    minimum_time_step=0.1,
                    )
        # plan path 
        time = crrt.plan_path()

        if time > self.max_planning_time:
            within_max_time = False
            print("CRRT EB time overflow: found time of ", time)
        time = min(time, self.max_planning_time)

        if (not crrt.path_found) or (not within_max_time):
            success=False

        highres_paths = dict(enumerate(crrt.get_high_resolution_paths()))
        paths, states, controls, timesteps, costs = crrt.get_path()

        if(self.printenv):
            pcol = ['xkcd:powder pink', 'xkcd:metallic blue', 'xkcd:pastel orange',
                    'xkcd:pastel blue', 'xkcd:terracotta', 'xkcd:purplish grey',
                    'xkcd:powder pink', 'xkcd:metallic blue', 'xkcd:pastel orange',
                    'xkcd:pastel blue', 'xkcd:terracotta', 'xkcd:purplish grey']
            if success:
                mprint = MultiRRTPrinter(env, crrt, paths, pcol, pcol, joint_states=True)
                mprint.print_rrt('crrt_eb_pipeline_' + str(seed) + "_" + str(len(agents)) + '.png',
                                 print_tree=False)
                mprint.print_highres_simulation(highres_paths,
                'crrt_eb_pipeline_' + str(seed) + "_" + str(len(agents)) + '.gif', animation_speed=100)


        if not success:
            return (False, time, 0, 0, 0, "") 

        if self.collision_checks:
            _raise_if_highres_paths_collide(
                crrt.get_high_resolution_path_numpy_array(),
                agent_objs,
                agent_objs[0].distance_metric_state_size,
                crrt.dynamic_agent_clearance,
                crrt.roundoff_digits)
            
        max_time = crrt.path_time
        costs = crrt.path_cost

        return (True, time, sum(costs), crrt.path_time, max_time, "")
    

class KinoTiCRRTEBTestClass(AbstractTestClass):
    """
    Test class for CRRT with Kino-TI EB
    """
    def __init__(self, printenv = False, collision_checks = False,
                 max_planning_time=300.0, print_logs = False,
                 debug_flag = False, obs_buffers = True,
                 truncate_paths = False, branch_goal_parking = True,
                 kd_delta_radius=0.1,
                 max_num_edges_per_node=1000,
                 num_extension_trials=1,
                 num_edge_candidates_per_agent=10,
                 max_joint_edge_trials=15,
                 epsilon_random=0.01,
                 fallback_to_random_control=True,
                 dynamic_agent_clearance=0.0):
        
        if branch_goal_parking and truncate_paths:
            raise ValueError(
                "branch_goal_parking requires truncate_paths=False")
        super().__init__(printenv, max_planning_time=max_planning_time,
                         print_logs=print_logs, debug_flag=debug_flag,
                         obs_buffers=obs_buffers)
        self.name = "CRRT with Kino-TI Edge Bundles"
        self.short_name = "K-TI EB CRRT"
        self.collision_checks = collision_checks
        self.truncate_paths = truncate_paths
        self.branch_goal_parking = branch_goal_parking
        self.kd_delta_radius = kd_delta_radius
        self.max_num_edges_per_node = max_num_edges_per_node
        self.num_extension_trials = num_extension_trials
        self.num_edge_candidates_per_agent = num_edge_candidates_per_agent
        self.max_joint_edge_trials = max_joint_edge_trials
        self.epsilon_random = epsilon_random
        self.fallback_to_random_control = fallback_to_random_control
        self.dynamic_agent_clearance = dynamic_agent_clearance

    """
    Each test class must have a similar 'test_func' that takes the same arguments 
    """
    def test_func(self, agents, starts, obstacles, goals, goal_radii,
                  seed, env_width, env_depth, env_height=None):
        print("Testing " + self.name)

        if env_height is None:
            env = SquareEnvironment(env_width, env_depth, obstacles,
                                    obs_buffers=self.obs_buffers)
        else:
            env = CuboidEnvironment(env_width, env_depth, env_height,
                            obstacles, obs_buffers=self.obs_buffers)
        
        agent_objs = []
        ebs = []
        isvalid_funcs = []
        cost_funcs = []
        random_pt_funcs = [] 
        reached_goal_funcs = [] 
        translate_funcs = [] 
        sort_edges_funcs = []
        eb_kd_trees = []
        get_eb_kd_tree_query_funcs = []
        sampling_time_steps = []
        within_max_time = True
        success=True

        for agent in agents:
            agent_obj = agent.get_agent()
            agent_objs.append(agent_obj)

            kino_ti_eb, kd_tree_ti_eb = agent.get_kino_ti_edge_bundle()
            ebs.append(kino_ti_eb)
            eb_kd_trees.append(kd_tree_ti_eb)
            
            isvalid_funcs.append(agent_obj.is_new_node_valid)
            cost_funcs.append(agent_obj.get_cost)
            random_pt_funcs.append(agent_obj.get_random_point)
            reached_goal_funcs.append(agent_obj.agent_reached_goal)
            translate_funcs.append(agent_obj.kd_tree_point_translate_function)
            if agent.sort_edges is True:
                sort_edges_funcs.append(agent_obj.sort_kd_tree_edges)
            else:
                sort_edges_funcs.append(agent_obj.no_sorting_kd_tree_edges)
            get_eb_kd_tree_query_funcs.append(agent_obj.get_eb_kd_tree_query)
            sampling_time_steps.append(agent.sampling_time_step)

        # instantiate Kite-CRRT obj
        crrt = KinoTIEBCRRT(agents=agent_objs, 
                    starts=starts,
                    goals=goals,
                    goal_radii=goal_radii,
                    edge_bundle=ebs,
                    env=env,
                    max_iter = np.inf, 
                    planning_time=self.max_planning_time,         
                    isvalid_function=isvalid_funcs, 
                    cost_function=cost_funcs,
                    random_point_function=random_pt_funcs, 
                    reached_goal_function = reached_goal_funcs,
                    translate_function=translate_funcs,
                    sort_edges_function=sort_edges_funcs,
                    udf_seed = seed, 
                    print_logs=self.print_logs,
                    debug_flag=self.debug_flag,
                    eb_kd_trees=eb_kd_trees,
                    get_eb_kd_tree_query_funcs=get_eb_kd_tree_query_funcs,
                    max_num_edges_per_node=self.max_num_edges_per_node,
                    num_extension_trials=self.num_extension_trials,
                    num_edge_candidates_per_agent=self.num_edge_candidates_per_agent,
                    max_joint_edge_trials=self.max_joint_edge_trials,
                    epsilon_random=self.epsilon_random,
                    fallback_to_random_control=self.fallback_to_random_control,
                    kd_tree_delta_radius=self.kd_delta_radius,
                    dynamic_agent_clearance=self.dynamic_agent_clearance,
                    truncate_paths=self.truncate_paths,
                    branch_goal_parking=self.branch_goal_parking,
                    use_fixed_sampling_time=False,
                    sampling_time_step=min(sampling_time_steps),
                    minimum_time_step=0.1, 
                    )
        # plan path 
        time = crrt.plan_path()

        if time > self.max_planning_time:
            within_max_time = False
            print("CRRT EB time overflow: found time of ", time)
        time = min(time, self.max_planning_time)

        if (not crrt.path_found) or (not within_max_time):
            success=False

        highres_paths = dict(enumerate(crrt.get_high_resolution_paths()))
        paths, states, controls, timesteps, costs = crrt.get_path()

        if(self.printenv):
            pcol = ['xkcd:powder pink', 'xkcd:metallic blue', 'xkcd:pastel orange',
                    'xkcd:pastel blue', 'xkcd:terracotta', 'xkcd:purplish grey',
                    'xkcd:powder pink', 'xkcd:metallic blue', 'xkcd:pastel orange',
                    'xkcd:pastel blue', 'xkcd:terracotta', 'xkcd:purplish grey']
            if success:
                mprint = MultiRRTPrinter(env, crrt, paths, pcol, pcol, joint_states=True)
                mprint.print_rrt('crrt_eb_pipeline_' + str(seed) + "_" + str(len(agents)) + '.png',
                                 print_tree=False)
                mprint.print_highres_simulation(highres_paths,
                'crrt_eb_pipeline_' + str(seed) + "_" + str(len(agents)) + '.gif', animation_speed=100)


        if not success:
            return (False, time, 0, 0, 0, "") 

        if self.collision_checks:
            _raise_if_highres_paths_collide(
                crrt.get_high_resolution_path_numpy_array(),
                agent_objs,
                agent_objs[0].distance_metric_state_size,
                crrt.dynamic_agent_clearance,
                crrt.roundoff_digits)
            
        max_time = crrt.path_time
        costs = crrt.path_cost

        # :MAINT: for cRRT-based planning, the max time and the average time
        # is ALWAYS the same, since cRRT plans for all agents simultaneously.
        return (True, time, sum(costs), crrt.path_time, max_time, "")
    

class PrrtTestClass(AbstractTestClass):
    """
    Test class for PRRT
    """
    def __init__(self, printenv = False, collision_checks = False,
                 max_planning_time=300.0, print_logs = False,
                 debug_flag = False, obs_buffers = True,
                 num_extension_trials=10,
                 goal_sampling_probability=0.01,
                 dynamic_agent_clearance=0.0):
        
        super().__init__(printenv, max_planning_time=max_planning_time,
                         print_logs=print_logs, debug_flag=debug_flag,
                         obs_buffers=obs_buffers)
        self.name = "PRRT"
        self.short_name = "PRRT"
        self.collision_checks = collision_checks
        self.num_extension_trials = num_extension_trials
        self.goal_sampling_probability = goal_sampling_probability
        self.dynamic_agent_clearance = dynamic_agent_clearance

    """
    Each test class must have a similar 'test_func' that takes the same arguments 
    """
    def test_func(self, agents, starts, obstacles, goals, goal_radii,
                  seed, env_width, env_depth, env_height=None):
        print("Testing " + self.name)

        if env_height is None:
            env = SquareEnvironment(env_width, env_depth, obstacles,
                                    obs_buffers=self.obs_buffers)
        else:
            env = CuboidEnvironment(env_width, env_depth, env_height,
                            obstacles, obs_buffers=self.obs_buffers)
        
        agent_objs = []
        isvalid_funcs = []
        cost_funcs = []
        random_pt_funcs = [] 
        reached_goal_funcs = [] 
        sampling_time_steps = []
        within_max_time = True
        success = True

        for agent in agents:
            agent_objs.append(agent.get_agent())
            isvalid_funcs.append(agent.get_valid_function())
            cost_funcs.append(agent.get_cost_function())
            random_pt_funcs.append(agent.get_random_pt_function())
            reached_goal_funcs.append(agent.get_reached_goal_function())
            sampling_time_steps.append(agent.sampling_time_step)

        planners = []
        seed_rng = np.random.default_rng(seed)
        planner_seeds = seed_rng.integers(0, np.iinfo(np.int32).max, size=len(agent_objs))
        sampling_time_step = min(sampling_time_steps)

        for i, (agent_obj, start, goal, goal_radius) in enumerate(zip(agent_objs, starts, goals, goal_radii)):
            env.add_agent(agent_obj, goal=(goal, goal_radius))
            planners.append(RRT(
                start=start,
                goal=goal,
                goal_radius=goal_radius,
                env=env,
                agent=agent_obj,
                use_fixed_sampling_time=False,
                sampling_time_step=sampling_time_step,
                minimum_time_step=0.1,
                max_iter=np.inf,
                planning_time=self.max_planning_time,
                num_extension_trials=self.num_extension_trials,
                isvalid_function=isvalid_funcs[i],
                cost_function=cost_funcs[i],
                random_point_function=random_pt_funcs[i],
                reached_goal_function=reached_goal_funcs[i],
                udf_seed=int(planner_seeds[i]),
                goal_sampling_probability=self.goal_sampling_probability,
                dynamic_agent_clearance=self.dynamic_agent_clearance,
                print_logs=self.print_logs,
                debug_flag=self.debug_flag,
            ))

        path_found, time, total_cost = PrioritizedPlanning.plan_multi(
            planners=planners,
            planning_time=self.max_planning_time,
            print_logs=self.print_logs)

        if time > self.max_planning_time:
            within_max_time = False
            print("PRRT time overflow: found time of ", time)
        time = min(time, self.max_planning_time)

        if (not path_found) or (not within_max_time):
            success=False

        if(self.printenv):
            pcol = ['xkcd:powder pink', 'xkcd:metallic blue', 'xkcd:pastel orange', 
                    'xkcd:pastel blue', 'xkcd:terracotta', 'xkcd:purplish grey',
                    'xkcd:powder pink', 'xkcd:metallic blue', 'xkcd:pastel orange',
                    'xkcd:pastel blue', 'xkcd:terracotta', 'xkcd:purplish grey'] * 3
            if success:
                paths = [planner.get_path()[0] for planner in planners]
                mprint = MultiRRTPrinter(env, planners, paths, pcol, pcol)
                mprint.print_rrt(self.savepath + 'prrt_pipeline' + str(seed) + "_" + str(len(agents)) + '.png',
                                 print_tree=False)
                highres_paths = dict(enumerate([planner.get_high_resolution_path() for planner in planners]))
                mprint.print_highres_simulation(highres_paths,
                self.savepath + 'prrt_pipeline' + str(seed) + "_" + str(len(agents)) + '.gif', animation_speed=100)

        if not success:
            return (False, time, 0, 0, 0, "") 

        if self.collision_checks:
            _raise_if_highres_paths_collide(
                [planner.get_high_resolution_path_numpy_array() for planner in planners],
                agent_objs,
                agent_objs[0].distance_metric_state_size,
                self.dynamic_agent_clearance,
                planners[0].roundoff_digits)
            
        path_times = []
        max_time = 0
        for planner in planners:
            path_times.append(planner.path_time)
            if planner.path_time > max_time:
                max_time = planner.path_time

        return (True, time, total_cost, np.average(path_times), max_time, "")
    

class PrrtEbTestClass(AbstractTestClass):
    """
    Test class for PRRT with edge bundles
    """
    def __init__(self, printenv = False, collision_checks = False,
                max_planning_time=300., print_logs = False,
                debug_flag = False, obs_buffers = True,
                dynamic_agent_clearance=0.0):
        
        super().__init__(printenv, max_planning_time=max_planning_time,
                        print_logs=print_logs, debug_flag=debug_flag,
                        obs_buffers=obs_buffers)
        self.name = "PRRT With Edge Bundles"
        self.short_name = "EB PRRT"
        self.collision_checks = collision_checks
        self.dynamic_agent_clearance = dynamic_agent_clearance

    """
    Each test class must have a similar 'test_func' that takes the same arguments 
    """
    def test_func(self, agents, starts, obstacles, goals, goal_radii,
                  seed, env_width, env_depth, env_height=None):
        print("Testing " + self.name)

        if env_height is None:
            env = SquareEnvironment(env_width, env_depth, obstacles,
                                    obs_buffers=self.obs_buffers)
        else:
            env = CuboidEnvironment(env_width, env_depth, env_height,
                            obstacles, obs_buffers=self.obs_buffers)
        
        agent_objs = []
        ebs = []
        isvalid_funcs = []
        cost_funcs = []
        random_pt_funcs = [] 
        reached_goal_funcs = [] 
        translate_funcs = [] 
        sort_edges_funcs = []
        sampling_time_steps = []
        nums_skip_edges = []
        within_max_time = True

        for agent in agents:
            agent_objs.append(agent.get_agent())
            ebs.append(agent.get_edge_bundle())
            isvalid_funcs.append(agent.get_valid_function())
            cost_funcs.append(agent.get_cost_function())
            random_pt_funcs.append(agent.get_random_pt_function())
            reached_goal_funcs.append(agent.get_reached_goal_function())
            translate_funcs.append(agent.get_point_translate_function())
            sort_edges_funcs.append(agent.get_sort_edges_func())
            sampling_time_steps.append(agent.sampling_time_step)
            nums_skip_edges.append(agent.num_skip_edges)

        (paths, states, rrts, controls, timesteps, time) = EdgeBundlePRRT.plan_multi(
                    agents=agent_objs, 
                    starts=starts,
                    goals=goals,
                    goal_radii=goal_radii,
                    env=env,
                    edge_bundle=ebs,
                    # change these after debug 
                    max_iter = np.inf, planning_time=self.max_planning_time,   
                    isvalid_function=isvalid_funcs, 
                    cost_function=cost_funcs,
                    random_point_function=random_pt_funcs, 
                    reached_goal_function = reached_goal_funcs,
                    translate_function=translate_funcs,
                    sort_edges_functions=sort_edges_funcs,
                    num_skip_edges=min(nums_skip_edges),
                    num_rand_edges=1,
                    udf_seed = seed,
                    use_fixed_sampling_time=False,
                    sampling_time_step=min(sampling_time_steps),
                    dynamic_agent_clearance=self.dynamic_agent_clearance,
                    print_logs=self.print_logs,
                    debug_flag=self.debug_flag
                    )
        if time > self.max_planning_time:
            within_max_time = False
            print("EB PRRT time overflow: found time of ", time)
        time = min(time, self.max_planning_time)

        # 'not rrts' will evaluate to True if empty, which will be true if not all agents
        # could find a path 
        if (not rrts) or (not within_max_time):
            return (False, time, 0, 0, 0, "")

        if(self.printenv):
            pcol = ['xkcd:powder pink', 'xkcd:metallic blue', 'xkcd:pastel orange',
                    'xkcd:pastel blue', 'xkcd:terracotta', 'xkcd:purplish grey']
            mprint = MultiRRTPrinter(env, rrts, paths, pcol, pcol)
            mprint.print_rrt('prrt_eb_pipeline' + str(seed) + '.png', print_tree=False)
            highres_paths = dict(enumerate([planner.get_high_resolution_path() for planner in rrts]))
            mprint.print_highres_simulation(highres_paths,
                        'prrt_eb_pipeline' + str(seed) + '.gif', animation_speed=100) 

        if self.collision_checks:
            _raise_if_highres_paths_collide(
                [planner.get_high_resolution_path_numpy_array() for planner in rrts],
                agent_objs,
                agent_objs[0].distance_metric_state_size,
                rrts[0].dynamic_agent_clearance,
                rrts[0].roundoff_digits)

        path_times = []
        max_time = 0
        costs = []
        for (path, rrt) in zip(paths, rrts):
            rrtNode = rrt.tree.nodes(data=True)[path[-1]].get('value')
            costs.append(rrtNode.cost_so_far)
            path_times.append(rrt.path_time)
            if rrt.path_time > max_time:
                max_time = rrt.path_time

        return (True, time, sum(costs), np.average(path_times), max_time, "")
    
    
class PrioritizedKinoTIRRTTestClass(AbstractTestClass):
    """
    Test class for Prioritized RRT with Kino-TI edge bundles
    """
    def __init__(self, printenv = False, collision_checks=False, 
                max_planning_time=300., print_logs = False,
                debug_flag = False, obs_buffers = True,
                kd_delta_radius=0.1,
                max_num_edges_per_node=1000,
                num_extension_trials=1, #num_random_edges for Kite (default is 1)
                epsilon_random=0.01,
                goal_sampling_probability=0.01,
                dynamic_agent_clearance=0.0,
                ):
        
        super().__init__(printenv, max_planning_time=max_planning_time,
                        print_logs=print_logs, debug_flag=debug_flag,
                        obs_buffers=obs_buffers)
        self.name = "Prioritized RRT with Kino-TI edge bundles"
        self.short_name = "KTI EB PRRT"
        self.kd_delta_radius = kd_delta_radius
        self.goal_sampling_probability = goal_sampling_probability
        self.dynamic_agent_clearance = dynamic_agent_clearance
        self.collision_checks = collision_checks
        self.max_num_edges_per_node = max_num_edges_per_node
        self.num_extension_trials = num_extension_trials
        self.epsilon_random = epsilon_random

    def test_func(self, agents, starts, obstacles, goals, goal_radii,
                  seed, env_width, env_depth, env_height=None):
        print("Testing " + self.name)

        if env_height is None:
            env = SquareEnvironment(env_width, env_depth, obstacles,
                                    obs_buffers=self.obs_buffers)
        else:
            env = CuboidEnvironment(env_width, env_depth, env_height,
                            obstacles, obs_buffers=self.obs_buffers)

        agent_objs = []
        planners = []
        for agent, start, goal, goal_radius in zip(agents, starts, goals, goal_radii):
            agent_obj = agent.get_agent()
            agent_objs.append(agent_obj)
            kino_ti_eb, kd_tree_ti_eb = agent.get_kino_ti_edge_bundle()
            if agent.sort_edges is True:
                sort_kd_func = agent_obj.sort_kd_tree_edges
            else:
                sort_kd_func = agent_obj.no_sorting_kd_tree_edges


            planners.append(KinoTIEBRRT( 
                    start=start, goal=goal,
                    goal_radius=goal_radius, 
                    env = env, agent=agent_obj,
                    edge_bundle = kino_ti_eb,
                    use_fixed_sampling_time=False,
                    sampling_time_step=agent.sampling_time_step,
                    minimum_time_step=0.1,
                    max_iter = np.inf,
                    planning_time=self.max_planning_time,
                    isvalid_function=agent_obj.is_new_node_valid,
                    cost_function=agent_obj.get_cost,
                    random_point_function=agent_obj.get_random_point,
                    reached_goal_function = agent_obj.agent_reached_goal,
                    translate_function = agent_obj.kd_tree_point_translate_function,
                    sort_edges_function=sort_kd_func,
                    max_num_edges_per_node=self.max_num_edges_per_node,
                    num_skip_edges= agent.num_skip_edges,
                    num_random_edges= self.num_extension_trials,
                    eb_kd_tree=kd_tree_ti_eb,
                    get_eb_kd_tree_query=agent_obj.get_eb_kd_tree_query,
                    kd_tree_delta_radius=self.kd_delta_radius,
                    udf_seed = agent.seed * seed,
                    goal_sampling_probability=self.goal_sampling_probability,
                    dynamic_agent_clearance=self.dynamic_agent_clearance,
                    debug_flag=self.debug_flag,
                    print_logs=self.print_logs,
                    )
                )

        path_found, time, cost = PrioritizedPlanning.plan_multi(
            planners=planners, planning_time=self.max_planning_time, print_logs=self.print_logs)
        if time > self.max_planning_time:
            print("pRRT Kino-TI EB: found time of ", time)
            path_found = False
        total_time = min(time, self.max_planning_time)

        path_times = []
        max_agent_path_time = 0
        for planner in planners:
            path_times.append(planner.path_time)
            if planner.path_time > max_agent_path_time:
                max_agent_path_time = planner.path_time

        if(self.printenv):
            planner_list = []
            paths_list = []

            # get paths for each agent 
            def enumerate_range(xs, start=0, step=.1):
                res = {}
                for x in xs:
                    res[start] = x
                    start += step
                    start = round(start, 1)
                return res
            highres_paths = (enumerate_range(planner.get_high_resolution_path_numpy_array()) for planner in planners)
            pcol = ['xkcd:powder pink', 'xkcd:metallic blue', 'xkcd:pastel orange',
                    'xkcd:pastel blue', 'xkcd:terracotta', 'xkcd:purplish grey'] * 5
            mprint = MultiRRTPrinter(env, planner_list, paths_list, pcol, pcol)
            if path_found:
                mprint.print_highres_simulation(dict(enumerate(highres_paths)),
                self.savepath + 'pRRT_kino-ti-eb_pipeline_' + str(seed) + "_" + str(len(agents)) + '.gif',
                animation_speed=100)

        if path_found == False:
            return (False, total_time, 0, 0, 0, "") 

        if self.collision_checks:
            _raise_if_highres_paths_collide(
                [planner.get_high_resolution_path_numpy_array() for planner in planners],
                agent_objs,
                agent_objs[0].distance_metric_state_size,
                self.dynamic_agent_clearance,
                planners[0].roundoff_digits)

        return (True, total_time, cost, np.average(path_times), max_agent_path_time, "")
    

class KcbsTestClass(AbstractTestClass):
    """
    Test class for KCBS with RRT
    """
    def __init__(self, printenv = False, collision_checks = False, 
                max_planning_time=300.0, print_logs = False,
                debug_flag = False, obs_buffers = True,
                num_low_level_planner_iterations=10000,
                num_extension_trials=10,
                dynamic_agent_clearance=0.0):
        
        super().__init__(printenv, max_planning_time=max_planning_time,
                         print_logs=print_logs, debug_flag=debug_flag,
                         obs_buffers=obs_buffers)
        self.name = "KCBS using RRT"
        self.short_name = "RRT KCBS"
        self.collision_checks = collision_checks
        self.num_low_level_planner_iterations = num_low_level_planner_iterations
        self.num_extension_trials = num_extension_trials
        self.dynamic_agent_clearance = dynamic_agent_clearance


    def test_func(self, agents, starts, obstacles, goals, goal_radii,
                  seed, env_width, env_depth, env_height=None):
        print("Testing " + self.name)

        if env_height is None:
            env = SquareEnvironment(env_width, env_depth, obstacles,
                                    obs_buffers=self.obs_buffers)
        else:
            env = CuboidEnvironment(env_width, env_depth, env_height,
                            obstacles, obs_buffers=self.obs_buffers)

        agent_objs = []
        planners = []
        for agent, start, goal, goal_radius in zip(agents, starts, goals, goal_radii):
            agent_obj = agent.get_agent()
            agent_objs.append(agent_obj)
            planners.append(ConstrainedRRT( 
                    start=start, goal=goal,
                    goal_radius=goal_radius, 
                    env = env, agent=agent_obj,
                    sampling_time_step=agent.sampling_time_step,
                    minimum_time_step=0.1,
                    max_iter = self.num_low_level_planner_iterations,
                    num_extension_trials=self.num_extension_trials,
                    planning_time=2*self.max_planning_time,        
                    isvalid_function=agent.get_valid_function(),
                    cost_function=agent.get_cost_function(),
                    random_point_function=agent.get_random_pt_function(), 
                    reached_goal_function = agent.get_reached_goal_function(),
                    udf_seed = agent.seed * seed,
                    use_fixed_sampling_time=False,
                    print_logs=self.print_logs,
                    debug_flag=self.debug_flag
                ))
        kcbs_planner = KCBS(
                    env = env,
                    agents = agent_objs,
                    low_level_planners = planners,
                    max_trials = np.inf,
                    planning_time = self.max_planning_time,
                    clearance_threshold=self.dynamic_agent_clearance,
                    print_logs=self.print_logs,
                    rng_seed=seed
                    )  
        path_found, paths, cost, time = kcbs_planner.plan_multi_agent_paths()
        message = "Conflict node count: " + str(kcbs_planner.node_list.count)
        if time > self.max_planning_time:
            print("KCBS time overflow: found time of ", time)
            path_found = False
        total_time = min(time, self.max_planning_time)

        if path_found == False:
            return (False, total_time, 0, 0, 0, message) 

        if self.collision_checks:
            _raise_if_highres_paths_collide(
                paths,
                agent_objs,
                agent_objs[0].distance_metric_state_size,
                kcbs_planner.clearance_threshold,
                kcbs_planner.roundoff_digits)

        path_times = []
        max_agent_path_time = 0
        for planner in planners:
            path_times.append(planner.path_time)
            if planner.path_time > max_agent_path_time:
                max_agent_path_time = planner.path_time

        if(self.printenv):
            planner_list = []
            paths_list = []

            # get paths for each agent 
            for i in range(len(agent_objs)):
                planner_list.append(planners[i])
                (ids, states, controls, timesteps) = planners[i].get_path()
                paths_list.append(ids)
            pcol = ['xkcd:powder pink', 'xkcd:metallic blue', 'xkcd:pastel orange',
                    'xkcd:pastel blue', 'xkcd:terracotta', 'xkcd:purplish grey',
                    'xkcd:powder pink', 'xkcd:metallic blue', 'xkcd:pastel orange',
                    'xkcd:pastel blue', 'xkcd:terracotta', 'xkcd:purplish grey']
            mprint = MultiRRTPrinter(env, planner_list, paths_list, pcol, pcol)
            # mprint.print_rrt('kcbs_pipeline' + str(seed) + "_" + str(len(agents)) + '.png', print_tree=False)
            if path_found:
                mprint.print_highres_simulation(paths,
                'kcbs_pipeline' + str(seed) + "_" + str(len(agents)) +'.gif',
                animation_speed=100)

        return (True, total_time, cost, np.average(path_times), max_agent_path_time, message)
    

class KcbsEbTestClass(AbstractTestClass):
    """
    Test class for KCBS using RRT with edge bundles
    """
    def __init__(self, printenv = False, collision_checks = False, 
                max_planning_time=300.,print_logs = False,
                debug_flag = False, obs_buffers = True,
                dynamic_agent_clearance=0.0):
        
        super().__init__(printenv, max_planning_time=max_planning_time,
                        print_logs=print_logs, debug_flag=debug_flag,
                        obs_buffers=obs_buffers)
        self.name = "KCBS using RRT With Type 2 Edge Bundles"
        self.short_name = "EB RRT KCBS"
        self.collision_checks = collision_checks
        self.dynamic_agent_clearance = dynamic_agent_clearance

    def test_func(self, agents, starts, obstacles, goals, goal_radii,
                  seed, env_width, env_depth, env_height=None):
        print("Testing " + self.name)
        
        if env_height is None:
            env = SquareEnvironment(env_width, env_depth, obstacles,
                                    obs_buffers=self.obs_buffers)
        else:
            env = CuboidEnvironment(env_width, env_depth, env_height,
                            obstacles, obs_buffers=self.obs_buffers)

        agent_objs = []
        planners = []
        for agent, start, goal, goal_radius in zip(agents, starts, goals, goal_radii):
            agent_obj = agent.get_agent()
            agent_objs.append(agent_obj)
            planners.append(ConstrainedEdgeBundleType2RRT( 
                    start=start, goal=goal,
                    goal_radius=goal_radius, 
                    env = env, agent=agent_obj,
                    edge_bundle=agent.get_edge_bundle(),
                    sampling_time_step=agent.sampling_time_step,
                    minimum_time_step=0.1,
                    max_iter = 10000,
                    num_random_edges= 1,
                    num_skip_edges= agent.num_skip_edges,
                    planning_time=2*self.max_planning_time,        
                    isvalid_function=agent.get_valid_function(),
                    cost_function=agent.get_cost_function(),
                    random_point_function=agent.get_random_pt_function(), 
                    reached_goal_function = agent.get_reached_goal_function(),
                    translate_function = agent.get_point_translate_function(),
                    sort_edges_function=agent.get_sort_edges_func(),
                    print_logs=self.print_logs,
                    debug_flag=self.debug_flag,
                    udf_seed = agent.seed * seed,
                    use_fixed_sampling_time=False)
                )

        kcbs_planner = KCBS(
                    env = env,
                    agents = agent_objs,
                    low_level_planners = planners,
                    max_trials = np.inf,
                    planning_time = self.max_planning_time,
                    clearance_threshold=self.dynamic_agent_clearance,
                    print_logs=self.print_logs,
                    rng_seed=seed
                    )  
        path_found, paths, cost, time = kcbs_planner.plan_multi_agent_paths()
        message = "Conflict node count: " + str(kcbs_planner.node_list.count)
        if time > self.max_planning_time:
            print("KCBS EB time overflow: found time of ", time)
            path_found = False
        total_time = min(time, self.max_planning_time)

        path_times = []
        max_agent_path_time = 0
        for planner in planners:
            path_times.append(planner.path_time)
            if planner.path_time > max_agent_path_time:
                max_agent_path_time = planner.path_time

        if(self.printenv):
            planner_list = []
            paths_list = []

            # get paths for each agent 
            for i in range(len(agent_objs)):
                planner_list.append(planners[i])
                (ids, states, controls, timesteps) = planners[i].get_path()
                paths_list.append(ids)
            pcol = ['xkcd:powder pink', 'xkcd:metallic blue', 'xkcd:pastel orange',
                    'xkcd:pastel blue', 'xkcd:terracotta', 'xkcd:purplish grey']
            mprint = MultiRRTPrinter(env, planner_list, paths_list, pcol, pcol)
            if path_found:
                mprint.print_highres_simulation(paths,
                'kcbs_pipeline_eb' + str(seed) + "_" + str(len(agents)) + '.gif',
                animation_speed=100)

        if path_found == False:
            return (False, total_time, 0, 0, 0, message) 

        if self.collision_checks:
            _raise_if_highres_paths_collide(
                paths,
                agent_objs,
                agent_objs[0].distance_metric_state_size,
                kcbs_planner.clearance_threshold,
                kcbs_planner.roundoff_digits)

        return (True, total_time, cost, np.average(path_times), max_agent_path_time, message)
    

class KcbsKinoTiEbTestClass(AbstractTestClass):
    """
    Test class for KCBS using RRT with kino-TI edge bundles
    """
    def __init__(self, printenv = False, collision_checks = False,
                max_planning_time=300.,print_logs = False,
                debug_flag = False, obs_buffers = True,
                kd_delta_radius=0.1,
                max_num_edges_per_node=1000,
                num_low_level_planner_iterations=10000,
                num_extension_trials=1, #num_random_edges for Kite (default is 1)
                epsilon_random=0.01,
                dynamic_agent_clearance=0.0):
        
        super().__init__(printenv, max_planning_time=max_planning_time,
                         print_logs=print_logs, debug_flag=debug_flag,
                         obs_buffers=obs_buffers)
        self.name = "KCBS using RRT With kino-TI Edge Bundles"
        self.short_name = "KTI EB RRT KCBS"
        self.collision_checks = collision_checks
        self.kd_delta_radius = kd_delta_radius
        self.num_low_level_planner_iterations = num_low_level_planner_iterations
        self.max_num_edges_per_node = max_num_edges_per_node
        self.num_extension_trials = num_extension_trials
        self.epsilon_random = epsilon_random
        self.dynamic_agent_clearance = dynamic_agent_clearance

    def test_func(self, agents, starts, obstacles, goals, goal_radii, 
                  seed, env_width, env_depth, env_height=None):
        print("Testing " + self.name)

        if env_height is None:
            env = SquareEnvironment(env_width, env_depth, obstacles,
                                    obs_buffers=self.obs_buffers)
        else:
            env = CuboidEnvironment(env_width, env_depth, env_height,
                            obstacles, obs_buffers=self.obs_buffers)

        agent_objs = []
        planners = []
        for agent, start, goal, goal_radius in zip(agents, starts, goals, goal_radii):
            agent_obj = agent.get_agent()
            agent_objs.append(agent_obj)
            kino_ti_eb, kd_tree_ti_eb = agent.get_kino_ti_edge_bundle()
            if agent.sort_edges is True:
                sort_kd_func = agent_obj.sort_kd_tree_edges
            else:
                sort_kd_func = agent_obj.no_sorting_kd_tree_edges


            planners.append(ConstrainedKinoTIEBRRT( 
                    start=start, goal=goal,
                    goal_radius=goal_radius,
                    env = env, agent=agent_obj, 
                    edge_bundle = kino_ti_eb,
                    use_fixed_sampling_time=False,
                    sampling_time_step=agent.sampling_time_step,
                    minimum_time_step=0.1,
                    max_iter = self.num_low_level_planner_iterations,
                    planning_time=2*self.max_planning_time,
                    isvalid_function=agent_obj.is_new_node_valid,
                    cost_function=agent_obj.get_cost,
                    random_point_function=agent_obj.get_random_point,
                    reached_goal_function = agent_obj.agent_reached_goal,
                    translate_function = agent_obj.kd_tree_point_translate_function,
                    sort_edges_function=sort_kd_func,
                    max_num_edges_per_node=self.max_num_edges_per_node,
                    num_skip_edges= agent.num_skip_edges,
                    epsilon_random=self.epsilon_random,
                    num_random_edges= self.num_extension_trials,
                    eb_kd_tree = kd_tree_ti_eb,
                    get_eb_kd_tree_query=agent_obj.get_eb_kd_tree_query,
                    kd_tree_delta_radius=self.kd_delta_radius,
                    udf_seed = 0, #Will be overwritten by KCBS init
                    debug_flag=False,
                    print_logs=False,)
                )

        kcbs_planner = KCBS(
                    env = env,
                    agents = agent_objs,
                    low_level_planners = planners,
                    max_trials = np.inf,
                    planning_time = self.max_planning_time,
                    clearance_threshold=self.dynamic_agent_clearance,
                    print_logs=self.print_logs,
                    rng_seed=seed
                    )  
        path_found, paths, cost, time = kcbs_planner.plan_multi_agent_paths()
        message = "Conflict node count: " + str(kcbs_planner.node_list.count)
        if time > self.max_planning_time:
            print("KCBS Kino-TI EB time overflow: found time of ", time)
            path_found = False
        total_time = min(time, self.max_planning_time)

        path_times = []
        max_agent_path_time = 0
        for planner in planners:
            path_times.append(planner.path_time)
            if planner.path_time > max_agent_path_time:
                max_agent_path_time = planner.path_time

        if(self.printenv):
            planner_list = []
            paths_list = []

            # get paths for each agent 
            for i in range(len(agent_objs)):
                planner_list.append(planners[i])
                (ids, states, controls, timesteps) = planners[i].get_path()
                paths_list.append(ids)
            pcol = ['xkcd:powder pink', 'xkcd:metallic blue', 'xkcd:pastel orange',
                    'xkcd:pastel blue', 'xkcd:terracotta', 'xkcd:purplish grey']
            mprint = MultiRRTPrinter(env, planner_list, paths_list, pcol, pcol)
            if path_found:
                mprint.print_highres_simulation(paths,
                'kcbs_pipeline_eb' + str(seed) + "_" + str(len(agents)) + '.gif',
                animation_speed=100)

        if path_found == False:
            return (False, total_time, 0, 0, 0, message) 

        if self.collision_checks:
            _raise_if_highres_paths_collide(
                paths,
                agent_objs,
                agent_objs[0].distance_metric_state_size,
                kcbs_planner.clearance_threshold,
                kcbs_planner.roundoff_digits)

        return (True, total_time, cost, np.average(path_times), max_agent_path_time, message)


class KcbsDbrrtTestClass(AbstractTestClass):
    """
    Test class for KCBS using RRT with kino-TI edge bundles
    """
    def __init__(self, printenv = False, collision_checks = False,
                max_planning_time=300., print_logs = False,
                debug_flag = False, obs_buffers = True,
                use_optimizer=True,
                num_low_level_planner_iterations=10000,
                max_candidate_motions_per_expand=1000,
                dbrrt_alpha=0.5,
                dbrrt_delta=0.3,
                cost_delta_factor=0.0,
                goal_sampling_probability=0.1,
                dynamic_agent_clearance=0.0):
        
        super().__init__(printenv, max_planning_time=max_planning_time,
                        print_logs=print_logs, debug_flag=debug_flag,
                        obs_buffers=obs_buffers)
        self.name = "KCBS using dbRRT"
        self.short_name = "dbRRT KCBS"
        self.collision_checks = collision_checks
        self.num_low_level_planner_iterations = num_low_level_planner_iterations
        self.max_candidate_motions_per_expand = max_candidate_motions_per_expand
        self.dbrrt_alpha = dbrrt_alpha
        self.dbrrt_delta = dbrrt_delta
        self.cost_delta_factor = cost_delta_factor
        self.goal_sampling_probability = goal_sampling_probability
        self.use_optimizer = use_optimizer
        self.dynamic_agent_clearance = dynamic_agent_clearance

    def _get_dbrrt_planner(self, agent, start, goal, goal_radius, env, agent_obj):
        motion_primitives, kd_tree = agent.get_dbrrt_motion_primitives()
        transform_function = agent.get_dbrrt_transform_function()

        planner = ConstrainedDbRRTPlanner(
            start=np.asarray(start, dtype=np.float64),
            goal=np.asarray(goal, dtype=np.float64),
            goal_radius=goal_radius,
            env=env,
            agent=agent_obj,
            motion_primitives=motion_primitives,
            alpha=self.dbrrt_alpha,
            delta=self.dbrrt_delta,
            minimum_time_step=0.1,
            max_iter=self.num_low_level_planner_iterations,
            planning_time=2*self.max_planning_time,
            isvalid_function=agent_obj.is_new_node_valid,
            cost_function=agent_obj.get_cost,
            random_point_function=agent_obj.get_random_point,
            reached_goal_function=agent_obj.agent_reached_goal,
            translate_function=agent_obj.kd_tree_point_translate_function,
            sort_edges_function=agent_obj.sort_kd_tree_edges,
            transform_trajectory_function=transform_function,
            motion_primitive_kd_tree=kd_tree,
            get_motion_primitive_kd_tree_query=agent_obj.get_eb_kd_tree_query,
            max_candidate_motions_per_expand=self.max_candidate_motions_per_expand,
            allow_intermediate_goal=True,
            cost_delta_factor=self.cost_delta_factor,
            goal_bias=self.goal_sampling_probability,
            goal_expand_mode="focused",
            random_expand_mode="randomized",
            dynamic_agent_clearance=self.dynamic_agent_clearance,
            udf_seed=0,  # Will be overwritten by KCBS init.
            debug_flag=False,
            print_logs=False,
        )
        if self.use_optimizer:
            planner.set_optimizer(agent.get_dbrrt_optimizer_function())
        return planner

    def test_func(self, agents, starts, obstacles, goals, goal_radii,
                  seed, env_width, env_depth, env_height=None):
        print("Testing " + self.name)

        if env_height is None:
            env = SquareEnvironment(env_width, env_depth, obstacles,
                                    obs_buffers=self.obs_buffers)
        else:
            env = CuboidEnvironment(env_width, env_depth, env_height,
                            obstacles, obs_buffers=self.obs_buffers)

        agent_objs = []
        planners = []
        for agent, start, goal, goal_radius in zip(agents, starts, goals, goal_radii):
            agent_obj = agent.get_agent()
            agent_objs.append(agent_obj)
            planners.append(self._get_dbrrt_planner(agent,
                start,goal,goal_radius,env,agent_obj))

        kcbs_planner = KCBS(
                    env = env,
                    agents = agent_objs,
                    low_level_planners = planners,
                    max_trials = np.inf,
                    planning_time = self.max_planning_time,
                    clearance_threshold=self.dynamic_agent_clearance,
                    print_logs=self.print_logs,
                    rng_seed=seed
                    )  
        path_found, paths, cost, time = kcbs_planner.plan_multi_agent_paths()
        message = "Conflict node count: " + str(kcbs_planner.node_list.count)
        if time > self.max_planning_time:
            print("KCBS idb-RRT time overflow: found time of ", time)
            path_found = False
        total_time = min(time, self.max_planning_time)

        path_times = []
        max_agent_path_time = 0
        for planner in planners:
            path_times.append(planner.path_time)
            if planner.path_time > max_agent_path_time:
                max_agent_path_time = planner.path_time

        if(self.printenv):
            planner_list = []
            paths_list = []

            # get paths for each agent 
            for i in range(len(agent_objs)):
                planner_list.append(planners[i])
                (ids, states, controls, timesteps) = planners[i].get_path()
                paths_list.append(ids)
            pcol = ['xkcd:powder pink', 'xkcd:metallic blue', 'xkcd:pastel orange',
                    'xkcd:pastel blue', 'xkcd:terracotta', 'xkcd:purplish grey']
            mprint = MultiRRTPrinter(env, planner_list, paths_list, pcol, pcol)
            if path_found:
                mprint.print_highres_simulation(paths,
                'kcbs_pipeline_eb' + str(seed) + "_" + str(len(agents)) + '.gif',
                animation_speed=100)

        if path_found == False:
            return (False, total_time, 0, 0, 0, message) 

        if self.collision_checks:
            _raise_if_highres_paths_collide(
                paths,
                agent_objs,
                agent_objs[0].distance_metric_state_size,
                kcbs_planner.clearance_threshold,
                kcbs_planner.roundoff_digits)

        return (True, total_time, cost, np.average(path_times), max_agent_path_time, message)


class DbCBSEnvTranslator(AbstractTestClass):
    """
    Translates envs into yaml for db-CBS 
    """
    def __init__(self, printenv = False, collision_checks = False,
                 max_planning_time=300.0, print_logs = False, debug_flag = False,
                 obs_buffers = True, save_location=""):
        
        super().__init__(printenv, max_planning_time=max_planning_time,
                         print_logs=print_logs, debug_flag=debug_flag,
                         obs_buffers=obs_buffers)
        self.name = "dbCBS Env Translator"
        self.short_name = "dbCBS Trans"
        self.collision_checks = collision_checks
        self.save_location = save_location + "/db_cbs_envs/" 
        os.makedirs(self.save_location, exist_ok=True) 


    def test_func(self, agents, starts, obstacles, goals, goal_radii,
                  seed, env_width, env_depth, env_height=None):
        print("Testing " + self.name)
        
        obs_to_export = []
        for obs in obstacles:
            if obs.shape == SquareEnvObsShape.RECTANGLE:
                center = [obs.x, obs.y]
                size = [obs.w, obs.h]
                obs_to_export.append({
                    'type': 'box',
                    'center': center,
                    'size': size
                    })
            elif obs.shape == SquareEnvObsShape.CUBOID:
                center = [obs.x, obs.y, obs.z]
                size = [obs.l, obs.w, obs.h]
                obs_to_export.append({
                    'type': 'box',
                    'center': center,
                    'size': size
                    })
            else:
                raise Exception("Only rectangle/cuboid obstacles are supported for db-CBS env export") 
        agents_to_export = []
        for agent, start, goal in zip(agents, starts, goals):
            if agent.dbcbs_name is None:
                raise Exception("Agent type " + str(type(agent)) + " does not have a db-CBS name specified for export")
            
            export_start = [float(val) for val in start]
            export_goal = [float(val) for val in goal]

            agents_to_export.append({
                'type': agent.dbcbs_name,
                'start': export_start,
                # :MAINT: our goals don't have an orientation, 
                # the start orientation seems as good a filler as any
                'goal': export_goal + export_start[(len(export_goal)):]
            })

        env_to_export = {
            'environment': {
                'max': [env_width, env_depth] + ([] if env_height is None else [env_height]),
                'min': [0., 0.] + ([] if env_height is None else [0.]),
                'obstacles': obs_to_export
            },
            'robots': agents_to_export
        }

        file_name = self.save_location + "dbcbs_env_" + str(len(agents)) + "_agents_seed_" + str(seed) + ".yaml"
        with open(file_name, 'w') as file:
            yaml.dump(env_to_export, file)
        
        return (False, 0, 0, 0, 0, "")
    
    def warmup_test(self, agents, starts, obstacles, goals, goal_radii,
                    seed, env_width, env_depth, env_height=None):
        """
        Warm up function to pre-compile numba functions not needed here!
        """
        return


class HimanshuEnvTranslator(AbstractTestClass):
    """
    Translates envs into python compaible with Himanshu's
    Test/Trial scripts  
    """
    def __init__(self, printenv = False, collision_checks = False,
                 max_planning_time=300.0, print_logs = False,
                 debug_flag = False, obs_buffers = True,
                 save_location=""):
        
        super().__init__(printenv, max_planning_time=max_planning_time,
                         print_logs=print_logs, debug_flag=debug_flag,
                         obs_buffers=obs_buffers)
        self.name = "Himanshu Env Translator"
        self.short_name = "Himanshu Trans"
        self.collision_checks = collision_checks
        self.save_location = save_location + "/himanshu_envs/" 
        os.makedirs(self.save_location, exist_ok=True) 

    def test_func(self, agents, starts, obstacles, goals, goal_radii,
                  seed, env_width, env_depth, env_height=None):
        print("Testing " + self.name) 

        with open(self.save_location + "himanshu_env_seed_" + str(seed) + ".py", 'w') as f:
            f.write("from Environments import *\n")
            f.write("from Agents import *\n\n")

            f.write("# seed: " + str(seed) + "\n\n")

            f.write("starts = [\n")
            for start in starts:
                f.write("\t(")
                for start_comp in start:
                    f.write(str(start_comp) + ", ")
                f.write("),\n")
            f.write("]\n\n")

            f.write("goals = [\n")
            for goal in goals:
                f.write("\t(")
                for goal_comp in goal:
                    f.write(str(goal_comp) + ", ")
                f.write("),\n")
            f.write("]\n\n")

            f.write("goal_radii = [\n")
            for goal_radius in goal_radii:
                f.write("\t" + str(goal_radius) + ",\n")
            f.write("]\n\n")

            f.write("obstacles = [\n")
            for obs in obstacles:
                if obs.shape == SquareEnvObsShape.CIRCLE:
                    f.write("\tCircleObstacle2D(" + str(obs.x) + ", "  + str(obs.y) + ", " + str(obs.r) + "),\n")
                elif obs.shape == SquareEnvObsShape.RECTANGLE:
                    f.write("\tRectangleObstacle2D(" + str(obs.x) + ", "  + str(obs.y) + ", " + str(obs.w) + ", " + str(obs.h) + "),\n")
                elif obs.shape == SquareEnvObsShape.CUBOID:
                    f.write(f"\tCuboidObstacle3D(x={obs.x}, y={obs.y}, z={obs.z}, l={obs.l}, w={obs.w}, h={obs.h}),\n")
            f.write("]\n\n")

            if env_height is None:
                f.write("env = SquareEnvironment(" + str(env_width) + ", " + str(env_depth) + ", obstacles)\n")
            else:
                f.write(f"env = CuboidEnvironment(length={env_width}, breadth={env_depth}, height={env_height}, obs=obstacles)\n")

            f.write("agents = [\n")
            for agent in agents:
                f.write("\t" + agent.get_agent_declaration() + ",\n")
            f.write("]\n")

        return (False, 0, 0, 0, 0, "")
    
    def warmup_test(self, agents, starts, obstacles, goals, goal_radii,
                    seed, env_width, env_depth, env_height=None):
        """
        Warm up function to pre-compile numba functions not needed here!
        """
        return
