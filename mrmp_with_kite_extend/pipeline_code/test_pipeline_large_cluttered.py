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
from printer import MultiRRTPrinter


class TestPipelineLarge(TestPipeline):
    def __init__(self, test_classes, agent_builders, test_rounds=100, num_agents=5, master_seed=42, 
                 env_width=40, env_bredth=40, savepath="", goal_radius=0.5, processes=1):
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
        if num_agents < 2: raise Exception("Use at least 2 agents")
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
            (8.475727025696276, 36.399396853155096, 2.351704883710214),
            (11.586823146518856, 36.0126973803567, 5.186804678458259),
            (21.60583263510259, 16.476363676107802, 0.38223657632653696),
            (21.38462394891068, 35.9108526038565, 5.585473581275507),
            (11.526827287177163, 28.961425844173046, 4.923798055189963),
            (17.154555241298887, 27.572396342307854, 4.059382377997044),
            (9.194008174486147, 32.51411098476325, 4.262651508019563),
            (3.0553400627138885, 10.412419495743238, 3.308785409838181),
            (8.419711803902793, 23.011167560297043, 1.8299192267904372),
            (35.61389396530191, 29.538268355135777, 0.46715533976667406),
            (27.97018128591573, 14.805134880139946, 2.5104753053093702),
            (34.228314106918226, 19.930206902949912, 1.8102825429964815),
            (13.936969738487413, 8.537920154746892, 6.229323587358334),
            (14.584178733413498, 19.492459300217995, 3.1329031605389273),
            (20.350369103540398, 13.060259182455626, 5.060143540072023),
            (5.350369103540398, 1.560259182455626, 5.060143540072023),
            (28.0553400627138885, 37.412419495743238, 3.308785409838181),
            (2.586823146518856, 26.0126973803567, 5.186804678458259),
            (27.154555241298887, 31.572396342307854, 4.059382377997044),
            (2.97018128591573, 14.805134880139946, 2.5104753053093702),
            (33.194008174486147, 6.51411098476325, 4.262651508019563),
            (24.0553400627138885, 30.412419495743238, 3.308785409838181),
            (16.61389396530191, 24.538268355135777, 0.46715533976667406),
            (10.226827287177163, 8.961425844173046, 4.923798055189963),
            (20.38462394891068, 19.9108526038565, 5.585473581275507),
            (26.936969738487413, 28.537920154746892, 6.229323587358334),
            (9.984178733413498, 16.892459300217995, 3.1329031605389273),
            (5.228314106918226, 22.930206902949912, 1.8102825429964815), 
            (14.60583263510259, 14.476363676107802, 0.38223657632653696),
            (13.475727025696276, 32.399396853155096, 2.351704883710214),
        ]

        possible_goals = [
            (34.864329915511654, 9.003899733940381),
            (27.300942040287637, 20.57283260237884),
            (21.415433893642362, 32.11505993112027),
            (18.965682757246398, 22.875128976549142),
            (15.782169681359699, 10.592860806584735),
            (23.895727973205105, 35.318459073195),
            (29.95421575545637, 31.261370762087413),
            (21.695409776980473, 21.46828957867312),
            (20.28401116779998, 34.36720330871995),
            (10.471055047841116, 33.422171568528704),
            (10.870944796891283, 12.280159780084496),
            (21.80886420867282, 28.720035908547196),
            (23.43037289856673, 26.50671273395829),
            (20.52385248009077, 4.957724964768499),
            (32.23290105920922, 9.441693400859048),
            (3.28401116779998, 18.36720330871995),
            (30.43037289856673, 24.50671273395829),
            (15.870944796891283, 16.480159780084496),
            (7.782169681359699, 25.592860806584735),
            (24.870944796891283, 14.280159780084496),
            (6.295409776980473, 9.46828957867312),
            (3.095727973205105, 37.318459073195),
            (37.80886420867282, 23.720035908547196),
            (36.300942040287637, 2.97283260237884),
            (31.471055047841116, 29.422171568528704),
            (9.965682757246398, 20.875128976549142),
            (27.415433893642362, 8.11505993112027),
            (14.55421575545637, 5.261370762087413),
            (4.52385248009077, 27.957724964768499),
            (4.864329915511654, 5.003899733940381),
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
            RectangleObstacle2D(7.2318180528742, 19.11557739153202, 3.6001340616706443, 1.2342962240137418),
            RectangleObstacle2D(24.640769120616746, 11.249958595699702, 1.8879733859231442, 2.7977578114173465),
            RectangleObstacle2D(33.52165105929789, 14.712475719357597, 4.987964054602642, 4.762326271741823),
            RectangleObstacle2D(25.087543392907982, 4.588990511682701, 4.740777979767133, 2.78139498421943),
            RectangleObstacle2D(33.89683742111731, 25.475322697342563, 1.933936687930509, 2.2835108061898968),
            RectangleObstacle2D(34.37105405176375, 35.2127489370062, 4.163599185649419, 2.673047891734387),
            RectangleObstacle2D(16.765313221135678, 33.66062958329129, 2.9450461561358456, 6.237881829980317),
            RectangleObstacle2D(4.578790030760515, 32.45567757895103, 2.1340532155846708, 2.612702802456554),
            RectangleObstacle2D(25.12892430122978, 22.266743199270557, 1.4311726211916573, 7.232857246763847),
            RectangleObstacle2D(13.385729240385057, 24.03349439022954, 1.3892207989803393, 3.6348281532725073),
            RectangleObstacle2D(6.971415847887632, 14.013040661417069, 3.8846399568092878, 2.352161305699923),
            RectangleObstacle2D(17.938939841099074, 6.780920639306122, 1.4716421879683201, 3.6851890360628623),
            RectangleObstacle2D(10.257196133603149, 5.4469093419979275, 4.115726446827529, 1.8541809184556888),
            RectangleObstacle2D(31.15259510533822, 4.60663958022095, 1.0702469277264908, 1.6214788027072329),
        ]

        # if (self.savepath != ""):
        #     filedir = self.savepath + "/env_checks"
        #     os.makedirs(filedir, exist_ok=True) 
        #     agent_objs = []
        #     for agent in agents:
        #         agent_objs.append(agent.get_agent())
        #     pcol = ['xkcd:powder pink', 'xkcd:metallic blue', 'xkcd:pastel orange', 'xkcd:pastel blue', 'xkcd:terracotta', 'xkcd:purplish grey',
        #                 'xkcd:powder pink', 'xkcd:metallic blue', 'xkcd:pastel orange', 'xkcd:pastel blue', 'xkcd:terracotta', 'xkcd:purplish grey'] * 5
        #     env = SquareEnvironment(self.env_width, self.env_bredth, obstacles, obs_buffers=False)
        #     MultiRRTPrinter.print_rrt_env(filedir + '/check_' + str(seed) + '.png',
        #                                         env, agent_objs, starts, goals, [self.goal_radius for _ in starts], pcol)

        # empty list is for obstacles
        return agents, starts, obstacles, goals, goal_radii
    

from test_classes import *
from agent_builders import *

if __name__ == "__main__":
    planning_time = 300.0  
    save_root = "test_results/new_final_results/large_cluttered_env"
    test_rounds = 100
    gr = 0.5
    kd_tree_delta_radius = .10
    seed_multiplier = 200
    num_processes = 25
    survival_min_successes = 1

    agent_builders = [
                    SecondOrderCarBuilder(),
                    UnicycleBuilder()
                    ]
    
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

    for agent_count in [4, 5, 8, 10, 15, 20, 25, 30]:
        master_seed = agent_count * seed_multiplier

        for agent_builder in agent_builders:
            # agent_count = 15

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
                                          branch_goal_parking=True) #, kd_delta_radius=kd_tree_delta_radius)
                    ]

            test_classes = filter_surviving_classes(test_classes, survival_key)
            if len(test_classes) == 0:
                print("No surviving classes for agent type", agent_builder.name, "with", agent_count, "agents. Skipping.")
                continue
            
            # test_classes = [DbCBSEnvTranslator(save_location=savepath)]

            print("Running tests for agent type", agent_builder.name, "with", agent_count, "agents and kd_tree_delta_radius", kd_tree_delta_radius)

            tp = TestPipelineLarge(test_classes, [agent_builder], test_rounds=test_rounds,
                                    num_agents=agent_count, 
                                    master_seed=master_seed,
                                    savepath=savepath, goal_radius=gr, processes=num_processes)
            write_pipeline_manifest(
                pipeline=tp,
                savepath=savepath,
                pipeline_file=__file__,
                environment_name="large_cluttered_env",
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
