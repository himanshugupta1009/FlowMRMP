import numpy as np
import yaml
import glob
import os

from test_pipeline_random import TestPipeline

import sys
sys.path.append('src')

from Environments import *
from utils import euclidean_distance


class TestPipelineDbaSwap(TestPipeline):
    OBS_KEY = 'obs'
    STARTS_KEY = 'starts'
    GOALS_KEY = 'goals'

    def __init__(self, test_classes, agent_builders, test_rounds=10, num_agents=8, master_seed=42):
        """
        Initialize a new test pipeline, based on https://github.com/IMRCLab/db-CBS/blob/main/example/gen_p10_n8_9_unicycle_sphere.yaml

        Args:
            test_classes: List of test classes (see below for examples)
            agent_builders (list(AgentBuilder)): List of agent builders to get 
                agents for test rounds 
            test_rounds (int, optional): Number of test rounds to run, where a random env is 
                created and run through each of the test classes. Defaults to 100.
            num_agents (int, optional): Number of agents in each round. Defaults to 8.
            master_seed (int, optional): Test pipeline rng seed. Defaults to 42.
        """
        if num_agents != 2: raise Exception("Use 2 agents for swap ONLY")

        path = 'pipeline_code/db_cbs_tests/'
        names = 'alcove_unicycle_sphere.yaml'
        files = glob.glob(os.path.join(path, names))

        self.envs = []
        self.env_counter = 0

        for filename in files:
            with open(filename, 'r') as file:
                content = file.read()
                filedata = yaml.safe_load(content)
                # print(filedata)

                this_env = {}
                obs = []
                for given_obs in filedata['environment']['obstacles']:
                    obs.append(RectangleObstacle2D(x=given_obs['center'][0], 
                                                   y=given_obs['center'][1], 
                                                   w=given_obs['size'][0], 
                                                   h=given_obs['size'][1]))
                this_env[TestPipelineDbaSwap.OBS_KEY] = obs

                starts = []
                goals = []
                for given_rob in filedata['robots']:
                    given_start = given_rob['start']
                    starts.append((given_start[0], given_start[1], given_start[2]))
                    given_goal = given_rob['goal']
                    goals.append((given_goal[0], given_goal[1]))

                this_env[TestPipelineDbaSwap.STARTS_KEY] = starts
                this_env[TestPipelineDbaSwap.GOALS_KEY] = goals 

                self.envs.append(this_env)  
                

        super().__init__(test_classes, agent_builders, test_rounds, num_agents, env_width=6.5, env_bredth=3.5, master_seed=master_seed)
        
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
        # from https://github.com/IMRCLab/db-CBS/blob/main/example/algorithms.yaml
        goal_radius = 0.2

        possible_starts = self.envs[self.env_counter][TestPipelineDbaSwap.STARTS_KEY]
        possible_goals = self.envs[self.env_counter][TestPipelineDbaSwap.GOALS_KEY]

        starts = []
        goals = []
        goal_radii = []
        # hardcoded params...for now
        goal_area = 0.

        agent_id_order = [i for i in range(len(agents))]
        np.random.default_rng(seed).shuffle(agent_id_order)
        print("Agent Order:", agent_id_order)

        for i in agent_id_order:
            starts.append(possible_starts[i])      
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

        obs = self.envs[self.env_counter][TestPipelineDbaSwap.OBS_KEY]
         
        agent_objs = []
        for agent in agents:
            agent_objs.append(agent.get_agent())
        pcol = ['xkcd:powder pink', 'xkcd:metallic blue', 'xkcd:pastel orange', 'xkcd:pastel blue', 'xkcd:terracotta', 'xkcd:purplish grey',
                    'xkcd:powder pink', 'xkcd:metallic blue', 'xkcd:pastel orange', 'xkcd:pastel blue', 'xkcd:terracotta', 'xkcd:purplish grey']
        env = SquareEnvironment(self.env_width, self.env_bredth, obs, obs_buffers=False)
        MultiRRTPrinter.print_rrt_env('checkalcove_' + str(self.num_agents) + '_' + str(self.env_counter) + '.png',
                                              env, agent_objs, starts, goals, goal_radii, pcol)

        self.env_counter += 1
        if self.env_counter >= len(self.envs): self.env_counter = 0
        
        return agents, starts, obs, goals, goal_radii
    
    

from test_classes import *
from agent_builders import *
    

# collect agent builders
# dynamics from https://github.com/quimortiz/dynobench/blob/05bafb374e5b00e858d351e2e89d8f4b409f56ab/models/unicycle_first_order_0_sphere.yaml
agent_builders = [UnicycleBuilder(max_speed=0.5, max_omega=2.0, radius=0.4, edge_bundle_file_location='edge_bundles/eb_unicycle_edges-1000_v05_av2.npz')]
max_time = 5.
test_classes = [KcbsEbTestClass(max_planning_time=max_time, obs_buffers=False), 
                KcbsTestClass(max_planning_time=max_time, obs_buffers=False), 
                PrrtEbTestClass(printenv=False, max_planning_time=max_time, obs_buffers=False), 
                PrrtTestClass(collision_checks=False,max_planning_time=max_time, obs_buffers=False),
                CRRTEBTestClass(printenv=False, max_planning_time=max_time, obs_buffers=False), 
                CRRTTestClass(printenv=False, max_planning_time=max_time,
                              obs_buffers=False, branch_goal_parking=True)]
tp = TestPipeline(test_classes, agent_builders, test_rounds=1, num_agents=8, master_seed=555)
tp.run()

for num_agents in [2]:
    # collect test classes
    max_time = 300.
    test_classes = test_classes = [KcbsEbTestClass(max_planning_time=max_time, obs_buffers=False), 
                KcbsTestClass(max_planning_time=max_time, obs_buffers=False), 
                PrrtEbTestClass(printenv=False, max_planning_time=max_time, obs_buffers=False), 
                PrrtTestClass(collision_checks=False,max_planning_time=max_time, obs_buffers=False),
                CRRTEBTestClass(printenv=num_agents==8, max_planning_time=max_time, obs_buffers=False), 
                CRRTTestClass(printenv=num_agents==8, max_planning_time=max_time,
                              obs_buffers=False, branch_goal_parking=True)]
    # instantiate test pipelines 
    tp = TestPipelineDbaSwap(test_classes, agent_builders, test_rounds=10, num_agents=num_agents, master_seed=num_agents*100)
    tp.run()
    tp.print_stats(filename="test_results/dbcbs_alcove_gr02", plots=True)
