import gc
import random 
import numpy as np
import os
from contextlib import redirect_stdout

from experiment_manifest import write_pipeline_manifest
from test_pipeline_random import TestPipeline

import sys
sys.path.append('src')

from Environments import *
from printer_3d import MultiRRTPrinter3d
from test_classes import *
from agent_builders import *

class TestPipelineOpen3d(TestPipeline):
    def __init__(self, test_classes, agent_builders, test_rounds=100, num_agents=5, master_seed=42, 
                 env_width=10., env_bredth=10., env_height=10., savepath="", goal_radius=0.5, processes=1,
                 save_env_checks=False):
        """
        Initialize a new test pipeline, based on the 'Open' environment from the 
        K-CBS paper

        Args:
            test_classes: List of test classes (see below for examples)
            agent_builders (list(AgentBuilder)): List of agent builders to get 
                agents for test rounds 
            test_rounds (int, optional): Number of test rounds to run, where a random env is 
                created and run through each of the test classes. Defaults to 100.
            num_agents (int, optional): Number of agents in each round. Defaults to 5.
            master_seed (int, optional): Test pipeline rng seed. Defaults to 42.
        """
        if num_agents < 2: raise Exception("Use at least 2 agents")
        super().__init__(test_classes, agent_builders, test_rounds, num_agents, env_width=env_width, env_bredth=env_bredth, env_height=env_height, 
                         master_seed=master_seed, savepath=savepath, goal_radius=goal_radius, processes=processes)
        self.save_env_checks = save_env_checks
        
    def wrap(self, value, max_val):
        return value % max_val

    def get_starts_goals(self, seed, agents):
        """
        Generates antipodal swap starts and goals on a sphere.
        """

        goal_radius = self.goal_radius
        N = self.num_agents
        num_swap_pairs = N // 2

        # Sphere radius safely inside environment bounds. With env_dim=10, this is 4.0.
        radius = min(self.env_width, self.env_bredth, self.env_height) / 2.0 - 1.0
        cx = self.env_width / 2.0
        cy = self.env_bredth / 2.0
        cz = self.env_height / 2.0

        # Fibonacci-style directions on one hemisphere. Each direction and its
        # antipode form an exact swap pair with straight-line distance 2 * radius.
        golden_angle = np.pi * (3.0 - np.sqrt(5.0))

        swap_pairs = []
        for i in range(num_swap_pairs):
            z = (i + 0.5) / num_swap_pairs
            r_xy = np.sqrt(max(0.0, 1.0 - z * z))
            phi = golden_angle * i

            ux = r_xy * np.cos(phi)
            uy = r_xy * np.sin(phi)
            uz = z

            plus = (cx + radius * ux, cy + radius * uy, cz + radius * uz)
            minus = (cx - radius * ux, cy - radius * uy, cz - radius * uz)
            swap_pairs.append((plus, minus))

        starts = []
        goals = []
        goal_radii = []
        goal_area = 0.0

        agent_id_order = list(range(len(agents)))
        # self.rng.shuffle(agent_id_order)
        print("Agent Order:", agent_id_order)

        extra_start = None
        extra_goal = None
        if N % 2 == 1:
            extra_angle = np.pi / 4.0
            extra_start = (
                cx + radius * np.cos(extra_angle),
                cy + radius * np.sin(extra_angle),
                cz,
            )
            extra_goal = (
                cx - radius * np.cos(extra_angle),
                cy - radius * np.sin(extra_angle),
                cz,
            )
            print("Odd agent count:", N)
            print("Base exact-swap agent count:", N - 1)
            print("Extra start:", extra_start)
            print("Extra goal:", extra_goal)

        for i in agent_id_order:
            if i < 2 * num_swap_pairs:
                plus, minus = swap_pairs[i // 2]
                if i % 2 == 0:
                    sx, sy, sz = plus
                    goal = minus
                else:
                    sx, sy, sz = minus
                    goal = plus
            else:
                sx, sy, sz = extra_start
                goal = extra_goal

            starts.append(
                agents[i].get_start(
                    0, 0, 0, 0, None,
                    x=sx,
                    y=sy,
                    z=sz,
                )
            )

            goals.append(goal)
            goal_radii.append(goal_radius)

        return starts, goals, goal_radii, goal_area

    def get_env_parms(self, seed): 
        """
        Gets all the necessary parameters to generate an environment 

        Args:
            seed (int): RNG seed

        Returns:
            agents, starts, obs_objs, goals, goal_radii lists for each agent
        """
        agents = self.get_agents(seed)
        starts, goals, goal_radii, goal_area = self.get_starts_goals(seed, agents)
        obstacles = []

        if self.save_env_checks and self.savepath != "":
            filedir = self.savepath + "/env_checks"
            os.makedirs(filedir, exist_ok=True) 
            agent_objs = []
            for agent in agents:
                agent_objs.append(agent.get_agent())
            pcol = ['xkcd:powder pink', 'xkcd:metallic blue', 'xkcd:pastel orange', 'xkcd:pastel blue', 'xkcd:terracotta', 'xkcd:purplish grey',
                'xkcd:powder pink', 'xkcd:metallic blue', 'xkcd:pastel orange', 'xkcd:pastel blue', 'xkcd:terracotta', 'xkcd:purplish grey']*3
            env = CuboidEnvironment(length=self.env_width, breadth=self.env_bredth, height=self.env_height, obs=obstacles)
            MultiRRTPrinter3d.print_rrt_env(filedir + '/check_' + str(seed) + '.png', 
                                            env, agents, starts, goals, goal_radii, pcol)
        # empty list is for obstacles
        return agents, starts, obstacles, goals, goal_radii

    
if __name__ == "__main__":
    agent_builders = [
        # QuadcopterBuilder(
        #     motion_primitive_file_location="motion_primitives/quadcopter6d_long_50_1000_primitives.npz",
        #     num_motion_primitives=1000,
        #     radius=0.3,),
        # QuadcopterBuilder(
        #     motion_primitive_file_location="motion_primitives/quadcopter6d_dbcbs_15_1100_primitives.npz",
        #     num_motion_primitives=1100,
        #     radius=0.3,),
        QuadcopterBuilder(
            motion_primitive_file_location="motion_primitives/quadcopter6d_long_50_max_length_6000_primitives.npz",
            num_motion_primitives=6000,
            radius=0.3,),
    ]
    planning_time = 300.0
    test_rounds = 100
    gr = 0.5
    optimizer_static_time_mode = "fixed_time"
    optimizer_constrained_time_mode = "fixed_time"
    optimizer_solver_ids = {
        "fixed_time": 0,
        "free_time": 1,
    }
    solver_id_static = optimizer_solver_ids[optimizer_static_time_mode]
    solver_id_constrained = optimizer_solver_ids[optimizer_constrained_time_mode]
    save_root = f"paper_results/results_9June2026/swap_3d_env/MP_6000_{optimizer_static_time_mode}/"
    # save_root = "paper_results/tests_kcbs_dbRRT_quad/MP_6000/AR_0p3_GR_0p5/free_time/swap_3d_env"
    # save_root = "paper_results/tests_kcbs_dbRRT_quad/MP_1100/AR_0p3_GR_0p5/free_time/swap_3d_env"
    kd_tree_delta_radius = .10
    seed_multiplier = 200
    survival_min_successes = 1
    env_dim = 10.0
    surviving_classes = {}
    surviving_classes[QuadcopterBuilder] = None

    def filter_surviving_classes(test_classes):
        if surviving_classes[QuadcopterBuilder] is None:
            surviving_classes[QuadcopterBuilder] = [tc.name for tc in test_classes]
            return test_classes

        return [tc for tc in test_classes if tc.name in surviving_classes[QuadcopterBuilder]]

    def get_failed_class_names(test_classes):
        failed = []
        for test_class in test_classes:
            successes = sum(bool(success) for success in test_class.success.values())
            if successes < survival_min_successes:
                failed.append(test_class.name)
        return failed

    def get_num_processes(agent_count):
        # if agent_count <= 10:
        #     return 100
        # if agent_count < 20:
        #     return 50
        return 15

    for agent_count in [2, 3, 4, 5, 8, 10, 12, 15, 18, 20, 23, 25, 27, 30]:
        master_seed = agent_count * seed_multiplier
        num_processes = get_num_processes(agent_count)

        for agent_builder in agent_builders:
            savepath = os.path.join(
                save_root,
                agent_builder.name
                + "_a" + str(agent_count)
                + "_tests" + str(test_rounds)
                + "_seed" + str(master_seed)
                + "_gr" + str(gr)
                + "_kd" + str(kd_tree_delta_radius)
            )
            os.makedirs(savepath, exist_ok=True)

            test_classes = [
                KcbsTestClass(max_planning_time=planning_time, obs_buffers=False), 
                KcbsKinoTiEbTestClass(max_planning_time=planning_time, obs_buffers=False),
                KcbsDbrrtTestClass(max_planning_time=planning_time, obs_buffers=False,
                    optimizer_backend="cpp_dynoplan",
                    cpp_optimizer_options=CppDynoplanQuadcopter6DOptimizerOptions(
                        solver_id_static=solver_id_static,
                        solver_id_constrained=solver_id_constrained,
                    ),
                ),
                PrrtTestClass(max_planning_time=planning_time, obs_buffers=False),
                PrioritizedKinoTIRRTTestClass(max_planning_time=planning_time, obs_buffers=False),
                CRRTTestClass(max_planning_time=planning_time,
                              obs_buffers=False,
                              branch_goal_parking=True),
                KinoTiCRRTEBTestClass(max_planning_time=planning_time,
                                      obs_buffers=False,
                                      branch_goal_parking=True),
                ]

            test_classes = filter_surviving_classes(test_classes)
            if len(test_classes) == 0:
                print("No surviving classes for agent type", agent_builder.name, "with", agent_count, "agents. Skipping.")
                continue
            # test_classes = [DbCBSEnvTranslator(save_location=savepath)]
            # test_classes = [HimanshuEnvTranslator(save_location=savepath)]

            print("Running tests for agent type", agent_builder.name, "with", agent_count, "agents.")

            tp = TestPipelineOpen3d(test_classes, [agent_builder], test_rounds=test_rounds, 
                                    num_agents=agent_count, master_seed=master_seed,
                                    env_width=env_dim, env_bredth=env_dim, env_height=env_dim,
                                    savepath=savepath, goal_radius=gr, processes=num_processes)
            extra_experiment_config = {
                "agent_type": agent_builder.name,
                "planning_time": planning_time,
                "kd_tree_delta_radius": kd_tree_delta_radius,
                "seed_multiplier": seed_multiplier,
                "survival_min_successes": survival_min_successes,
                "num_processes": num_processes,
                "env_dim": env_dim,
            }
            if isinstance(agent_builder, QuadcopterBuilder):
                extra_experiment_config.update({
                    "dbrrt_optimizer_backend": "cpp_dynoplan",
                    "dbrrt_optimizer_static_time_mode": optimizer_static_time_mode,
                    "dbrrt_optimizer_constrained_time_mode": optimizer_constrained_time_mode,
                    "dbrrt_solver_id_static": solver_id_static,
                    "dbrrt_solver_id_constrained": solver_id_constrained,
                })
            write_pipeline_manifest(
                pipeline=tp,
                savepath=savepath,
                pipeline_file=__file__,
                environment_name="swap_3d_env",
                extra_experiment_config=extra_experiment_config,
            )
            with open(savepath+'/log.txt', 'w') as f, redirect_stdout(f):
                tp.run()
            stats_failed = tp.print_stats(filename=savepath, plots=True)
            all_failed = get_failed_class_names(tp.test_classes)
            if set(stats_failed) != set(all_failed):
                print("Stats failed classes:", stats_failed)
            print("All Failed Classes:", all_failed)

            # Update surviving classes
            surviving_classes[QuadcopterBuilder] = [
                class_name for class_name in surviving_classes[QuadcopterBuilder]
                if class_name not in all_failed
            ]
            print("Surviving Quadcopter Classes:", surviving_classes[QuadcopterBuilder])
            gc.collect()
