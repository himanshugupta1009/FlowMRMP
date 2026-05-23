from test_pipeline_random import *

import sys
sys.path.append('src')

from Environments import *

"""
WARNINGS BLOCK
"""
# import warnings

# def warn_with_traceback(message, category, filename, lineno, file=None, line=None):
#     log_file = file if file is not None else sys.stderr
#     log_file.write(warnings.formatwarning(message, category, filename, lineno, line))
#     traceback.print_stack(file=log_file)

# warnings.showwarning = warn_with_traceback
"""
WARNINGS BLOCK
"""

class TestPipelineSinglenv(TestPipeline):
        
    def get_starts_goals(self, seed, agents):
        
        starts = [
            (11.777928026849953, 21.59844486823614, 4.88647878507452),
            (20.88221806694752, 18.5715364546508, 5.783653477198367),
            (17.26787681391891, 24.577219097361247, 1.634977677442681),
            (30.351033533894803, 16.537015427676415, 4.4588686087289835),
            (4.562214631169679, 28.35277618607915, 5.33199369149357),
            (18.01110024280572, 7.534413109091251, 4.766415496383577),
            (27.468922231056446, 26.57728684704878, 5.108978544587677),
            (6.608877736613589, 10.015873303463604, 0.8007753424826467),
            (35.83657083137417, 14.528914971558782, 0.626384426708539),
            (27.095496298989172, 22.913389810443295, 3.5204270508218616),
        ]

        goals = [
            (26.888782417316712, 17.7429599279794),
            (29.873658660226127, 13.849995177191571),
            (23.048259578106435, 7.184855222188487),
            (10.176878071346723, 25.16868907134616),
            (17.09627683864005, 22.514479251911347),
            (20.2184709196353, 33.95194160838833),
            (12.675392792153557, 33.24418608632648),
            (24.603200318900925, 27.670466695857932),
            (25.13025406716617, 11.38210327043382),
            (9.380839186810054, 19.52126983951552),
        ]

        goal_radii = [
            2.0,
            2.0,
            2.0,
            2.0,
            2.0,
            2.0,
            2.0,
            2.0,
            2.0,
            2.0,
        ]

        return starts, goals, goal_radii, 0
    
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
            RectangleObstacle2D(11.8757112695261, 14.7753869011411, 2.9222297732556215, 3.773504415643333),
            RectangleObstacle2D(26.454113809683605, 34.006665905957064, 2.061124225305981, 6.292944782355627),
            RectangleObstacle2D(30.909268092072253, 4.238809607434206, 3.3991570907837527, 2.274400260287219),
            RectangleObstacle2D(35.63329077135981, 32.553480390964495, 3.215440308252057, 3.716070416877762),
            RectangleObstacle2D(11.386398975871206, 6.4401604230387735, 1.0914669614143433, 5.296017252359851),
            RectangleObstacle2D(9.326812710136899, 36.07996378662771, 2.1516863565546345, 1.4279793502613005),
            RectangleObstacle2D(36.352041672106566, 25.97874105018752, 1.730096554018254, 1.1463871077243368),
        ]

        # end loop over obs creation 
        return agents, starts, obstacles, goals, [self.goal_radius for _ in starts] #goal_radii  


from test_classes import *
from agent_builders import *

if __name__ == "__main__":
    # warm up numba
    planning_time = 5.0 
    test_classes = [KcbsEbTestClass(max_planning_time=planning_time), 
                    KcbsTestClass(max_planning_time=planning_time), 
                    KcbsKinoTiEbTestClass(max_planning_time=planning_time),
                    PrrtEbTestClass(max_planning_time=planning_time), 
                    PrrtTestClass(max_planning_time=planning_time),
                    CRRTEBTestClass(max_planning_time=planning_time), 
                    CRRTTestClass(max_planning_time=planning_time,
                                  branch_goal_parking=True),
                    PrioritizedKinoTIRRTTestClass(max_planning_time=planning_time),
                    KinoTiCRRTEBTestClass(max_planning_time=planning_time,
                                          branch_goal_parking=True)
                    ]
    # collect agent builders
    agent_builders = [UnicycleBuilder(max_speed=0.5, max_omega=2.0, radius=0.4, \
                                       edge_bundle_file_location='edge_bundles/eb_unicycle_edges-1000_v05_av2.npz',
                                       kino_ti_edge_bundle_file_location='edge_bundles/eb_unicycle_dbcbs_kinodynamic_TI_edges_100000.npz')]
    # instantiate test pipeline 
    tp = TestPipelineSinglenv(test_classes, agent_builders, test_rounds=1, num_agents=2, master_seed=50)
    tp.run()

    planning_time = 300.0
    num_agents = 10
    for gr in [2.]:
        savepath = "test_results/ourrandom_test_results/SINGLE_ENV_a" + str(num_agents) + "_gr" + str(gr)
        os.makedirs(savepath, exist_ok=True) 
        # collect test classes
        test_classes =  [KcbsEbTestClass(max_planning_time=planning_time), 
                KcbsTestClass(max_planning_time=planning_time), 
                KcbsKinoTiEbTestClass(max_planning_time=planning_time),
                PrrtEbTestClass(max_planning_time=planning_time), 
                PrrtTestClass(max_planning_time=planning_time),
                PrioritizedKinoTIRRTTestClass(max_planning_time=planning_time),
                CRRTEBTestClass(max_planning_time=planning_time), 
                CRRTTestClass(max_planning_time=planning_time,
                              branch_goal_parking=True),
                KinoTiCRRTEBTestClass(max_planning_time=planning_time,
                                      branch_goal_parking=True)]


        # instantiate test pipeline 
        tp = TestPipelineSinglenv(test_classes, agent_builders, test_rounds=100, num_agents=num_agents, master_seed=num_agents*100+5, goal_radius=gr,
                            savepath=savepath, obstacle_classes=[RectangleObstacle2D])
        # with open(savepath+'/log.txt', 'w') as f, redirect_stdout(f):
        tp.run()
        tp.print_stats(filename=savepath, plots=True)
