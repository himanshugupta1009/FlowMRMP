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
from agent_builders import QuadcopterBuilder


class TestPipeline3d(TestPipeline):
    def __init__(self, test_classes, agent_builders=[QuadcopterBuilder()], test_rounds=100, num_agents=5, master_seed=42, 
                 env_width=14.0, env_bredth=14.0, env_height=10., savepath="", goal_radius=0.3, processes=1, obs=True):
        """
        Initialize a new test pipeline for a static 3d env. 

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
        if num_agents not in range(2, 31): raise Exception("Use 2 - 30 agents")
        self.obs = obs
        super().__init__(test_classes, agent_builders, test_rounds, num_agents, env_width=env_width, env_bredth=env_bredth, env_height=env_height,
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
            (1.0, 1.0, 0.6, 0.0, 0.0, 0.0), 
            (13.0, 1.0, 0.6, 0.0, 0.0, 0.0), 
            (1.0, 13.0, 0.6, 0.0, 0.0, 0.0), 
            (13.0, 13.0, 0.6, 0.0, 0.0, 0.0), 
            (7.0, 1.0, 0.6, 0.0, 0.0, 0.0), 
            (7.0, 3.0, 0.6, 0.0, 0.0, 0.0),
            (2.5, 6.5, 0.5, 0.0, 0.0, 0.0),
            (11.5, 10.5, 1.0, 0.0, 0.0, 0.0),
            (5.0, 2.0, 0.6, 0.0, 0.0, 0.0),
            (10.0, 5.0, 1.5, 0.0, 0.0, 0.0),
            (7.0, 10.0, 0.8, 0.0, 0.0, 0.0),
            (2.0, 10.0, 1.0, 0.0, 0.0, 0.0),
            (12.0, 5.0, 2.0, 0.0, 0.0, 0.0),
            (9.0, 1.5, 0.7, 0.0, 0.0, 0.0),
            (3.5, 9.0, 1.2, 0.0, 0.0, 0.0),
            (10.5, 12.0, 1.5, 0.0, 0.0, 0.0),
            (5.5, 5.5, 0.9, 0.0, 0.0, 0.0),
            (12.5, 7.2, 2.5, 0.0, 0.0, 0.0),
            (2.0, 3.0, 1.5, 0.0, 0.0, 0.0),
            (8.5, 10.0, 2.0, 0.0, 0.0, 0.0),
            (4.5, 7.0, 1.0, 0.0, 0.0, 0.0),
            (7.5, 12.0, 2.5, 0.0, 0.0, 0.0),
            (11.0, 9.0, 0.6, 0.0, 0.0, 0.0),
            (1.5, 3.0, 5.0, 0.0, 0.0, 0.0),
            (9.5, 2.5, 1.8, 0.0, 0.0, 0.0),
            (6.5, 2.0, 0.7, 0.0, 0.0, 0.0),
            (3.0, 13.0, 1.5, 0.0, 0.0, 0.0),
            (10.5, 8.5, 1.2, 0.0, 0.0, 0.0),
            (5.5, 11.5, 2.0, 0.0, 0.0, 0.0),
            (9.0, 6.0, 0.8, 0.0, 0.0, 0.0),
        ]

        possible_goals = [
            (13.0, 13.0, 8.6), 
            (1.0, 13.0, 3.6), 
            (13.0, 1.0, 7.6), 
            (1.0, 1.0, 1.6), 
            (7.0, 13.0, 5.6), 
            (7.0, 7.0, 8.6),
            (11.0, 2.0, 2.5),
            (2.5, 3.5, 0.8),
            (12.5, 11.5, 8.0),
            (1.5, 7.5, 4.0),
            (2.0, 1.5, 3.0),
            (10.0, 1.0, 5.5),
            (1.5, 12.0, 7.0),
            (12.0, 6.5, 3.5),
            (8.5, 1.5, 6.5),
            (3.0, 12.5, 2.5),
            (13.0, 8.0, 4.0),
            (4.5, 1.5, 7.0),
            (12.0, 12.5, 5.0),
            (5.0, 0.8, 8.5),
            (6.5, 12.0, 3.5),
            (12.5, 4.5, 1.5),
            (2.8, 2.5, 7.5),
            (10.0, 11.0, 2.0),
            (1.0, 9.5, 6.0),
            (13.0, 5.0, 8.5),
            (3.0, 10.0, 1.5),
            (11.5, 1.8, 4.0),
            (7.5, 3.5, 9.0),
            (2.5, 8.0, 2.5),
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
            starts.append(ns)  
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
            CuboidObstacle3D(x=5.288024544853885, y=10.428115301068528, z=4.381429795846468, l=1.3839478994290928, w=0.5401805203430455, h=8.762859591692935),
            CuboidObstacle3D(x=11.954231894339882, y=8.603275488906887, z=4.1909151213488585, l=0.4894908302066708, w=1.3916708816110792, h=8.381830242697717),
            CuboidObstacle3D(x=5.09798694910239, y=4.379210742901877, z=3.5329163210542545, l=1.012947629187785, w=0.8241902727000516, h=7.065832642108509),
            CuboidObstacle3D(x=1.2515176919283175, y=6.382882438156907, z=3.7064760014154894, l=0.8154560303422158, w=0.9315903102362977, h=7.412952002830979),
            CuboidObstacle3D(x=3.571110230587391, y=11.42056387486772, z=4.2468278944017905, l=1.3569195226036317, w=0.5305939467851933, h=8.493655788803581),
            CuboidObstacle3D(x=7.747853019130037, y=6.206600327515435, z=4.375561998755278, l=0.7725502893336461, w=1.0734098954898244, h=8.751123997510556),
            CuboidObstacle3D(x=9.369922496110094, y=3.789125351106821, z=3.866387903731101, l=1.41759692208027, w=0.494146098178711, h=7.732775807462202),
            CuboidObstacle3D(x=7.572100162570637, y=8.433738463586065, z=3.7151803959960374, l=0.6944531823772292, w=0.9656203291792224, h=7.430360791992075),
            CuboidObstacle3D(x=1.3788626583269576, y=4.612971585766165, z=3.602016866375745, l=0.6801371563340349, w=1.0735310416558315, h=7.20403373275149),
            CuboidObstacle3D(x=1.2517230753901782, y=8.230925811928927, z=3.8643541654261577, l=0.8436815330362557, w=0.6464442743787236, h=7.7287083308523155),
            CuboidObstacle3D(x=11.937082922873042, y=3.321434501007907, z=4.427651086838168, l=0.8820119319327673, w=1.0452295541498402, h=8.855302173676336),
            CuboidObstacle3D(x=2.6144639370162315, y=4.4039025815115345, z=3.2908979727461443, l=0.5014458528208032, w=1.3934776335531336, h=6.5817959454922885),
            CuboidObstacle3D(x=8.882920490259526, y=11.497996916324377, z=3.6938262682356733, l=0.4695094664176922, w=1.3328617718899038, h=7.387652536471347),
            CuboidObstacle3D(x=10.771924060959083, y=6.681457358183268, z=3.947871746185684, l=0.4353548933573186, w=1.1882000449382848, h=7.895743492371368),
            CuboidObstacle3D(x=5.156669442822354, y=9.227777527413963, z=4.005500757253257, l=0.9586972526631968, w=1.021113964861763, h=8.011001514506514),
            CuboidObstacle3D(x=3.654977319031023, y=2.4409283243404722, z=3.7095992809212595, l=0.5059145301349421, w=1.061011454249298, h=7.419198561842519),

        ]

        if not self.obs: obstacles = []

        # empty list is for obstacles
        return agents, starts, obstacles, goals, goal_radii
    

from test_classes import *
from agent_builders import *

if __name__ == "__main__":
    agent_builders = [QuadcopterBuilder()]
    planning_time = 300.0
    save_root = "test_results/new_final_results/large_cluttered_3d_env"
    test_rounds = 100
    gr = 0.3
    kd_tree_delta_radius = .10
    seed_multiplier = 200
    num_processes = 30
    survival_min_successes = 1
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

    for agent_count in [2, 3, 4, 5, 6, 8, 10, 15, 20, 25, 30]:
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
                # DbCBSEnvTranslator(save_location=savepath)
            ]

            test_classes = filter_surviving_classes(test_classes)
            if len(test_classes) == 0:
                print("No surviving classes for agent type", agent_builder.name, "with", agent_count, "agents. Skipping.")
                continue

            print("Running tests for agent type", agent_builder.name, "with", agent_count, "agents.")

            tp = TestPipeline3d(test_classes, [agent_builder], test_rounds=test_rounds, 
                                num_agents=agent_count, master_seed=master_seed,
                                savepath=savepath, goal_radius=gr, processes=num_processes)
            write_pipeline_manifest(
                pipeline=tp,
                savepath=savepath,
                pipeline_file=__file__,
                environment_name="large_cluttered_3d_env",
                extra_experiment_config={
                    "agent_type": agent_builder.name,
                    "planning_time": planning_time,
                    "kd_tree_delta_radius": kd_tree_delta_radius,
                    "seed_multiplier": seed_multiplier,
                    "survival_min_successes": survival_min_successes,
                    "obs": tp.obs,
                },
            )
            with open(savepath+'/log.txt', 'w') as f, redirect_stdout(f):
                tp.run()
            stats_failed = tp.print_stats(filename=savepath, plots=True)
            all_failed = get_failed_class_names(tp.test_classes)
            if set(stats_failed) != set(all_failed):
                print("Stats failed classes:", stats_failed)
            print("All Failed Classes:", all_failed)

            surviving_classes[QuadcopterBuilder] = [
                class_name for class_name in surviving_classes[QuadcopterBuilder]
                if class_name not in all_failed
            ]
            print("Surviving Quadcopter Classes:", surviving_classes[QuadcopterBuilder])
            gc.collect()
