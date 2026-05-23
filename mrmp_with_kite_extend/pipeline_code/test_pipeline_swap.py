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
        thetas = [ (2 * np.pi / self.num_agents) * i for i in range(self.num_agents) ]
        radius = ((self.env_width + self.env_bredth) / 2) / 2 - 1.0
        possible_starts = [ (self.env_width/2 + radius * np.cos(theta), 
                             self.env_bredth/2 + radius * np.sin(theta)) for theta in thetas]

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
            goal_index = round(self.wrap(i + self.num_agents//2, self.num_agents))
            goals.append(possible_starts[goal_index])
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
                        SecondOrderCarBuilder(),
                        UnicycleBuilder()
                      ]
    planning_time = 300.0
    save_root = "test_results/new_final_results/swap_env"
    test_rounds = 100
    gr = 0.5
    kd_tree_delta_radius = .10
    seed_multiplier = 200
    num_processes = 25
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

    for agent_count in [3, 4, 5, 8, 10, 15, 20, 25, 30]:
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
            write_pipeline_manifest(
                pipeline=tp,
                savepath=savepath,
                pipeline_file=__file__,
                environment_name="swap_env",
                extra_experiment_config={
                    "agent_type": agent_builder.name,
                    "planning_time": planning_time,
                    "kd_tree_delta_radius": kd_tree_delta_radius,
                    "seed_multiplier": seed_multiplier,
                    "survival_min_successes": survival_min_successes,
                    "env_dim": env_dim,
                },
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
