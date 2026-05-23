import gc
import random 
import numpy as np
import os
from contextlib import redirect_stdout

from test_pipeline_random import TestPipeline
from experiment_manifest import write_pipeline_manifest

import sys
sys.path.append('src')

from Environments import *
from printer import MultiRRTPrinter


class TestPipelineCluttered(TestPipeline):
    def __init__(self, test_classes, agent_builders, test_rounds=100, num_agents=5, master_seed=42, 
                 env_width=15, env_bredth=15, savepath="", goal_radius=0.2, processes=1):
        """
        Initialize a new test pipeline, based on the 'Large' environment from the 
        K-CBS paper

        Args:
            test_classes: List of test classes (see below for examples)
            agent_builders (list(AgentBuilder)): List of agent builders to get 
                agents for test rounds 
            test_rounds (int, optional): Number of test rounds to run, where a random env is 
                created and run through each of the test classes. Defaults to 100.
            num_agents (int, optional): Number of agents in each round. Defaults to 5.
            master_seed (int, optional): Test pipeline rng seed. Defaults to 42.
            processes (int, optional): Number of parallel processes to use. Defaults to 1.
        """
        if num_agents not in [x for x in range(2, 21)]: raise Exception("Use between 2 and 10 agents")
        super().__init__(test_classes, agent_builders, test_rounds, num_agents, env_width=env_width, env_bredth=env_bredth, 
                         master_seed=master_seed, savepath=savepath, goal_radius=goal_radius, processes=processes)
    
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

        possible_starts = [
            (1.0, 1.0, 0.0),    # red
            (6.0, 5.0, 0.0),    # teal
            (12.0, 5.0, 0.0),   # blue
            (14.0, 4.0, 0.0),   # green
            (9.0, 10.0, 0.0),   # magenta
            (3.0, 11.0, 0.0),   # red/orange
            (6.0, 14.0, 0.0),   # green (top)
            (8.0, 5.0, 0.0),    # purple
            (10.0, 1.0, 0.0),   # yellow
            (11.0, 14.0, 0.0),  # blue (top)
            (13.0, 10.0, 0.0),    
            (4., 8., 0.0),
            (9.0, 7.0, 0.0),
            (8.0, 12.0, 0.0),
            (1.0, 6.0, 0.0),
            (14.0, 12.0, 0.0),
            (4.0, 3.0, 0.0),
            (14.0, 1.0, 0.0),
            (6.0, 8.0, 0.0),
            (7.0, 1.0, 0.0)
        ]

        possible_goals = [
            (6.0, 3.0),     # red
            (2.0, 11.0),    # teal
            (5.0, 12.0),    # blue
            (9.0, 1.0),     # green
            (14.0, 3.0),    # magenta
            (12.0, 12.0),   # red/orange
            (3.0, 7.0),     # green (top)
            (13.0, 6.0),    # purple
            (11.0, 8.0),    # yellow
            (3.0, 14.0),    # blue (top)
            (12.0, 3.0),
            (1., 3.),
            (5.0, 9.0),
            (4.0, 5.0),
            (10.0, 6.0),
            (8.0, 3.0),
            (6.0, 11.0),
            (9.0, 8.0),
            (10.0, 12.0), 
            (3.0, 9.0)
        ]

        starts = []
        goals = []
        goal_radii = []
        # hardcoded params...for now
        goal_area = 0.
        agent_id_order = [i for i in range(len(agents))]
        # uncomment below to shuffle agent order
        # self.rng.shuffle(agent_id_order)
        print("Agent Order:", agent_id_order)

        for i in agent_id_order:
            ns = possible_starts[i]
            starts.append(agents[i].get_start(self.env_width, self.env_bredth, 0, self.rng, x=ns[0], y=ns[1], t=ns[2]))  
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
        obstacles = [
            # bottom row
            RectangleObstacle2D(4.5, 1.5, 3, 1),
            RectangleObstacle2D(12.5, 1.5, 1, 1),

            # lower-middle
            RectangleObstacle2D(2.0, 4.5, 2, 1),
            RectangleObstacle2D(10.0, 3.5, 2, 1),

            # middle
            RectangleObstacle2D(.5, 7.5, 1, 1),
            RectangleObstacle2D(1.5, 9.5, 1, 1),
            RectangleObstacle2D(13.5, 8.0, 1, 2),

            # vertical center column
            RectangleObstacle2D(7.5, 8.0, 1, 4),

            # upper-middle
            RectangleObstacle2D(2.5, 12.5, 3, 1),
            RectangleObstacle2D(10.5, 10.5, 1, 1),

            # top
            RectangleObstacle2D(7.5, 14.5, 1, 1),
            RectangleObstacle2D(13.0, 14.0, 2, 2),
        ]

        if (self.savepath != ""):
            filedir = self.savepath + "/env_checks"
            os.makedirs(filedir, exist_ok=True) 
            agent_objs = []
            for agent in agents:
                agent_objs.append(agent.get_agent())
            pcol = ['xkcd:powder pink', 'xkcd:metallic blue', 'xkcd:pastel orange', 'xkcd:pastel blue', 'xkcd:terracotta', 'xkcd:purplish grey',
                        'xkcd:powder pink', 'xkcd:metallic blue', 'xkcd:pastel orange', 'xkcd:pastel blue', 'xkcd:terracotta', 'xkcd:purplish grey'] * 5
            env = SquareEnvironment(self.env_width, self.env_bredth, obstacles, obs_buffers=False)
            # MultiRRTPrinter.print_rrt_env(filedir + '/check_' + str(seed) + '.png',
            #                                     env, agent_objs, starts, goals, [self.goal_radius for _ in starts], pcol)

        # empty list is for obstacles
        return agents, starts, obstacles, goals, goal_radii
    

from test_classes import *
from agent_builders import *

if __name__ == "__main__":
    agent_builders = [
                        SecondOrderCarBuilder(),
                        UnicycleBuilder()
                      ]
    planning_time = 300.0    
    save_root = "test_results/new_final_results/small_cluttered_env"
    test_rounds = 100
    gr = 0.5
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

    for agent_count in [3, 4, 5, 6, 7, 8, 9, 10, 12, 15, 18, 20]:
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
                test_classes = [
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
                    # KcbsEbTestClass(max_planning_time=planning_time, obs_buffers=False),
                    # PrrtEbTestClass(max_planning_time=planning_time, obs_buffers=False),
                    # CRRTEBTestClass(max_planning_time=planning_time, obs_buffers=False)
                    ]
            else:
                test_classes = [
                    KcbsTestClass(max_planning_time=planning_time, obs_buffers=False), 
                    KcbsKinoTiEbTestClass(max_planning_time=planning_time, obs_buffers=False), # kd_delta_radius=kd_tree_delta_radius),
                    PrrtTestClass(max_planning_time=planning_time, obs_buffers=False),
                    PrioritizedKinoTIRRTTestClass(max_planning_time=planning_time, obs_buffers=False), # kd_delta_radius=kd_tree_delta_radius),
                    CRRTTestClass(max_planning_time=planning_time,
                                  obs_buffers=False,
                                  branch_goal_parking=True),
                    KinoTiCRRTEBTestClass(max_planning_time=planning_time,
                                          obs_buffers=False,
                                          branch_goal_parking=True,
                                          print_logs=False) #, kd_delta_radius=kd_tree_delta_radius)
                    ]

            test_classes = filter_surviving_classes(test_classes, survival_key)
            if len(test_classes) == 0:
                print("No surviving classes for agent type", agent_builder.name, "with", agent_count, "agents. Skipping.")
                continue

            # test_classes = [DbCBSEnvTranslator(save_location=savepath)]

            print("Running tests for agent type", agent_builder.name, "with", agent_count, "agents and kd_tree_delta_radius", kd_tree_delta_radius)

            tp = TestPipelineCluttered(test_classes, [agent_builder], test_rounds=test_rounds,
                                    num_agents=agent_count, 
                                    master_seed=master_seed,
                                    savepath=savepath, goal_radius=gr, processes=num_processes)
            write_pipeline_manifest(
                pipeline=tp,
                savepath=savepath,
                pipeline_file=__file__,
                environment_name="small_cluttered_env",
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
