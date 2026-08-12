import gc
import os
import random 
import numpy as np
import numpy.ma as ma
import traceback

from experiment_manifest import write_pipeline_manifest
from test_pipeline_random import TestPipeline

import sys
sys.path.append('src')

from Environments import *
from utils import euclidean_distance


class TestPipelineOpen(TestPipeline):
    def __init__(self, test_classes, agent_builders, test_rounds=100, num_agents=5, master_seed=42, 
                 env_width=30, env_bredth=30, savepath="", goal_radius=0.5, processes=1):
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
        super().__init__(test_classes, agent_builders, test_rounds, num_agents, env_width=env_width, env_bredth=env_bredth, 
                         master_seed=master_seed, savepath=savepath, goal_radius=goal_radius, processes=processes)
        
    def wrap(self, value, max_val):
        return value % max_val

    def get_starts_goals(self, seed, agents):
        """
        Generates random starts/goals that aren't too
        close to one another

        Args:
            seed (int): pseudorandom number generator seed
            agents: agent objects

        Returns:
            list(agent_state_type), list(tuple(float, float)), list(float), float: 
                starts, goals, goal_radii, goal_area
        """
        goal_radius = self.goal_radius
        radius = ((self.env_width + self.env_bredth) / 2) / 2 - 1.0
        center = (self.env_width / 2, self.env_bredth / 2)
        num_even_agents = self.num_agents if self.num_agents % 2 == 0 else self.num_agents - 1
        thetas = [(2 * np.pi / num_even_agents) * i for i in range(num_even_agents)]
        possible_starts = [(center[0] + radius * np.cos(theta),
                            center[1] + radius * np.sin(theta)) for theta in thetas]
        possible_goals = list(possible_starts)

        if self.num_agents % 2 == 1:
            extra_theta = np.pi / num_even_agents
            extra_start = (center[0] + radius * np.cos(extra_theta),
                           center[1] + radius * np.sin(extra_theta))
            extra_goal = (center[0] - radius * np.cos(extra_theta),
                          center[1] - radius * np.sin(extra_theta))
            possible_starts.append(extra_start)
            possible_goals.append(extra_goal)
            thetas.append(extra_theta)

        starts = []
        goals = []
        goal_radii = []
        # hardcoded params...for now
        goal_area = 0.
        agent_id_order = [i for i in range(len(agents))]
        # self.rng.shuffle(agent_id_order)
        print("Agent Order:", agent_id_order)

        for i in agent_id_order:
            starts.append(agents[i].get_start(0, 0, 0, None, x=possible_starts[i][0], y=possible_starts[i][1], t = self.wrap(thetas[i]+np.pi, 2*np.pi)))  
            if i < num_even_agents:
                goal_index = round(self.wrap(i + num_even_agents//2, num_even_agents))
                goals.append(possible_goals[goal_index])
            else:
                goals.append(possible_goals[i])
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

        # if (self.savepath != ""):
        #     filedir = self.savepath + "/env_checks"
        #     os.makedirs(filedir, exist_ok=True) 
        #     agent_objs = []
        #     for agent in agents:
        #         agent_objs.append(agent.get_agent())
        #     pcol = ['xkcd:powder pink', 'xkcd:metallic blue', 'xkcd:pastel orange', 'xkcd:pastel blue', 'xkcd:terracotta', 'xkcd:purplish grey',
        #                 'xkcd:powder pink', 'xkcd:metallic blue', 'xkcd:pastel orange', 'xkcd:pastel blue', 'xkcd:terracotta', 'xkcd:purplish grey'] * 5
        #     env = SquareEnvironment(self.env_width, self.env_bredth, [], obs_buffers=False)
        #     MultiRRTPrinter.print_rrt_env(filedir + '/check_' + str(seed) + '.png',
        #                                         env, agent_objs, starts, goals, [self.goal_radius for _ in starts], pcol)

        # empty list is for obstacles
        return agents, starts, obstacles, goals, goal_radii


from test_classes import *
from agent_builders import *
from contextlib import redirect_stdout
    
if __name__ == "__main__":
    agent_builders = [
                        # SecondOrderCarBuilder(),
                        UnicycleBuilder()
                      ]
    planning_time = 300.0
    save_root = "paper_results/results_9June2026/swap_env"
    # save_root = "paper_results/debug/swap_env_soc/"
    # save_root = "paper_results/free_time/swap_env"
    test_rounds = 100
    gr = 0.5
    kd_tree_delta_radius = .10
    seed_multiplier = 100 #I used 100 for Unicycle and 200 for SOC
    survival_min_successes = 1
    env_dim = 20

    surviving_classes = {}
    surviving_classes[SecondOrderCarBuilder] = None
    surviving_classes[UnicycleBuilder] = None

    def get_survival_key(agent_builder):
        if isinstance(agent_builder, UnicycleBuilder):
            return UnicycleBuilder
        return SecondOrderCarBuilder

    def filter_surviving_classes(test_classes, survival_key):
        if surviving_classes[survival_key] is None:
            surviving_classes[survival_key] = [tc.name for tc in test_classes]
            return test_classes

        return [tc for tc in test_classes if tc.name in surviving_classes[survival_key]]

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
        # return 25
        return 10

    for agent_count in [2, 3, 4, 5, 8, 10, 12, 15, 18, 20, 23, 25, 27, 30]:
        master_seed = agent_count * seed_multiplier
        num_processes = get_num_processes(agent_count)
        # num_processes = 50 if agent_count < 18 else 25

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

            test_classes = []
            survival_key = get_survival_key(agent_builder)
            if isinstance(agent_builder, UnicycleBuilder):
                test_classes = [
                    KcbsTestClass(max_planning_time=planning_time, obs_buffers=False,
                        goal_sampling_probability=0.01),
                    KcbsKinoTiEbTestClass(max_planning_time=planning_time, obs_buffers=False,
                        goal_sampling_probability=0.01),
                    KcbsDbrrtTestClass(max_planning_time=planning_time, obs_buffers=False,
                        goal_sampling_probability=0.01,
                        optimizer_backend="cpp_dynoplan",
                        cpp_optimizer_options=CppDynoplanUnicycleOptimizerOptions(
                            solver_id_static=1,
                            solver_id_constrained=0,
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
                    # KcbsEbTestClass(max_planning_time=planning_time, obs_buffers=False),
                    # PrrtEbTestClass(max_planning_time=planning_time, obs_buffers=False),
                    # CRRTEBTestClass(max_planning_time=planning_time, obs_buffers=False)
                    ]
            else:
                test_classes = [
                    KcbsTestClass(max_planning_time=planning_time, obs_buffers=False), 
                    KcbsKinoTiEbTestClass(max_planning_time=planning_time, obs_buffers=False),
                    PrrtTestClass(max_planning_time=planning_time, obs_buffers=False),
                    PrioritizedKinoTIRRTTestClass(max_planning_time=planning_time, obs_buffers=False),
                    CRRTTestClass(max_planning_time=planning_time,
                                  obs_buffers=False,
                                  branch_goal_parking=True),
                    KinoTiCRRTEBTestClass(max_planning_time=planning_time,
                                          obs_buffers=False,
                                          branch_goal_parking=True,
                                          print_logs=False)
                    ]

            test_classes = filter_surviving_classes(test_classes, survival_key)
            if len(test_classes) == 0:
                print("No surviving classes for agent type", agent_builder.name, "with", agent_count, "agents. Skipping.")
                continue
            
            # test_classes = [DbCBSEnvTranslator(save_location=savepath)]
            # test_classes = [HimanshuEnvTranslator(save_location=savepath)]

            print("Running tests for agent type", agent_builder.name, "with", agent_count, "agents.")

            tp = TestPipelineOpen(test_classes, [agent_builder], test_rounds=test_rounds, num_agents=agent_count, 
                                    master_seed=master_seed, env_width=env_dim, env_bredth=env_dim,
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
            if isinstance(agent_builder, UnicycleBuilder):
                extra_experiment_config.update({
                    "dbrrt_optimizer_backend": "cpp_dynoplan",
                    "dbrrt_optimizer_static_time_mode": "free_time",
                    "dbrrt_optimizer_constrained_time_mode": "fixed_time",
                    "dbrrt_solver_id_static": 1,
                    "dbrrt_solver_id_constrained": 0,
                })
            write_pipeline_manifest(
                pipeline=tp,
                savepath=savepath,
                pipeline_file=__file__,
                environment_name="swap_env",
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
            if survival_key is UnicycleBuilder:
                surviving_classes[UnicycleBuilder] = [
                    class_name for class_name in surviving_classes[UnicycleBuilder]
                    if class_name not in all_failed
                ]
                print("Surviving Unicycle Classes:", surviving_classes[UnicycleBuilder])
            else:
                surviving_classes[SecondOrderCarBuilder] = [
                    class_name for class_name in surviving_classes[SecondOrderCarBuilder]
                    if class_name not in all_failed
                ]
                print("Surviving SecondOrderCar Classes:", surviving_classes[SecondOrderCarBuilder])

            gc.collect()
