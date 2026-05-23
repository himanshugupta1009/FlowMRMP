import gc
import numpy as np
import os
from contextlib import redirect_stdout

from experiment_manifest import write_pipeline_manifest
from test_pipeline_random import TestPipeline

import sys
sys.path.append('src')

from Environments import *
from utils import euclidean_distance


class TestPipelineCorridor(TestPipeline):
    def __init__(self, test_classes, agent_builders, test_rounds=100, num_agents=3, master_seed=42, 
                 goal_radius=0.25, processes=1, savepath=""):
        """
        Initialize a new test pipeline, based on the 'corridor' environment from the 
        K-CBS paper

        Args:
            test_classes: List of test classes (see below for examples)
            agent_builders (list(AgentBuilder)): List of agent builders to get 
                agents for test rounds 
            test_rounds (int, optional): Number of test rounds to run, where a random env is 
                created and run through each of the test classes. Defaults to 100.
            num_agents (int, optional): Number of agents in each round. Defaults to 3.
            env_width (int, optional): Env 'x value.' Defaults to 40.
            env_bredth (int, optional): Env 'y value.' Defaults to 40.
            master_seed (int, optional): Test pipeline rng seed. Defaults to 42.
        """
        if num_agents > 3 or num_agents < 0: raise Exception("Up to 4 agents allowed")
        super().__init__(test_classes, agent_builders, test_rounds, num_agents, env_width=5, env_bredth=4, master_seed=master_seed, 
                         goal_radius=goal_radius, processes=processes, savepath=savepath)
        
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

        loc_rng = np.random.default_rng(seed * 67)

        goal_radius = self.goal_radius
        possible_starts = [(3.5, 3.5),(1.0, 1.5),(4.0, 1.5)]
        possible_goals =  [(0.5, 2.5), (4.5, 1.5),(2.5, 1.5)]
        possible_thetas = [np.pi, 0, np.pi]

        starts = []
        goals = []
        goal_radii = []
        # hardcoded params...for now
        goal_area = 0.

        agent_id_order = [i for i in range(len(agents))]
        loc_rng.shuffle(agent_id_order) # himanshu approved
        print("Agent Order:", agent_id_order)

        for i in agent_id_order:
            starts.append(agents[i].get_start(0, 0, 0, None, x=possible_starts[i][0], y=possible_starts[i][1], t = possible_thetas[i]))      
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

        obs = []

        obs.append(RectangleObstacle2D(x = 1.5, y=0.5, w=3, h=1))
        obs.append(RectangleObstacle2D(x = 2.8, y=2.5, w=3.6, h=1))
         
        # empty list is for obstacles
        return agents, starts, obs, goals, goal_radii
    
    

from test_classes import *
from agent_builders import *

if __name__ == "__main__":
    agent_builders = [
                        SecondOrderCarBuilder(),
                        UnicycleBuilder()
                      ]
    planning_time = 300.0
    save_root = "test_results/new_final_results/narrow_corridor_env"
    test_rounds = 100
    gr = 0.25
    kd_tree_delta_radius = .10
    seed_multiplier = 200
    num_processes = 25
    survival_min_successes = 1

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

    for agent_count in [3]:
        master_seed = agent_count * seed_multiplier

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
                test_classes =  [
                    KcbsTestClass(max_planning_time=planning_time, obs_buffers=False), 
                    KcbsKinoTiEbTestClass(max_planning_time=planning_time, obs_buffers=False),
                    KcbsDbrrtTestClass(max_planning_time=planning_time, obs_buffers=False),
                    PrrtTestClass(max_planning_time=planning_time, obs_buffers=False),
                    PrioritizedKinoTIRRTTestClass(max_planning_time=planning_time, obs_buffers=False),
                    CRRTTestClass(max_planning_time=planning_time,
                                  obs_buffers=False,
                                  branch_goal_parking=True),
                    KinoTiCRRTEBTestClass(max_planning_time=planning_time,
                                          obs_buffers=False,
                                          branch_goal_parking=True), 
                    # DbCBSEnvTranslator(save_location=savepath)
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

            print("Running tests for agent type", agent_builder.name, "with", agent_count, "agents.")

            tp = TestPipelineCorridor(test_classes, [agent_builder], test_rounds=test_rounds, 
                                    num_agents=agent_count, 
                                    master_seed=master_seed,
                                    savepath=savepath,
                                    goal_radius=gr, processes=num_processes)
            write_pipeline_manifest(
                pipeline=tp,
                savepath=savepath,
                pipeline_file=__file__,
                environment_name="narrow_corridor_env",
                extra_experiment_config={
                    "agent_type": agent_builder.name,
                    "planning_time": planning_time,
                    "kd_tree_delta_radius": kd_tree_delta_radius,
                    "seed_multiplier": seed_multiplier,
                    "survival_min_successes": survival_min_successes,
                },
            )
            with open(savepath+'/log.txt', 'w') as f, redirect_stdout(f):
                tp.run()
            stats_failed = tp.print_stats(filename=savepath, plots=True)
            all_failed = get_failed_class_names(tp.test_classes)
            if set(stats_failed) != set(all_failed):
                print("Stats failed classes:", stats_failed)
            print("All Failed Classes:", all_failed)

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
