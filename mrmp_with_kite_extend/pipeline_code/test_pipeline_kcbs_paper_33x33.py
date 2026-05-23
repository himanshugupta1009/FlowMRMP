import numpy as np
import numpy.ma as ma

import sys
sys.path.append('src')

from Environments import *
from test_pipeline_random import TestPipeline

class TestPipelinePaperRandom(TestPipeline):
    def __init__(self, test_classes, agent_builders, test_rounds=100, num_agents=10, env_width=33, env_bredth=33, master_seed=42, num_obs = 0,
                 obstacle_types = [RectangleObstacle2D, CircularObstacle2D]):
        """
        Initialize a new test pipeline

        Args:
            test_classes: List of test classes (see below for examples)
            agent_builders (list(AgentBuilder)): List of agent builders to get 
                agents for test rounds 
            test_rounds (int, optional): Number of test rounds to run, where a random env is 
                created and run through each of the test classes. Defaults to 100.
            num_agents (int, optional): Number of agents in each round. Defaults to 5.
            env_width (int, optional): Env 'x value.' Defaults to 40.
            env_bredth (int, optional): Env 'y value.' Defaults to 40.
            master_seed (int, optional): Test pipeline rng seed. Defaults to 42.
            num_obs: (int, optional): number of obstacles to generate. Defaults to 0.
            obstacle_types: (list[AbstractObstacle], optional): Obstacle types to use. 
        """
        super().__init__(test_classes, agent_builders, test_rounds=test_rounds, num_agents=num_agents, 
                         env_width=env_width, env_bredth=env_bredth, master_seed=master_seed)

        self.num_obs = num_obs
        self.obstacle_types = obstacle_types
        
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
        loc_rng = np.random.default_rng(seed)

        starts = []
        goals = []
        goal_radii = []
        goal_buffer = 1
        # hardcoded params...for now
        goal_radius = 0.5
        env_start_end_buffer = 0.75
        goal_area = 0.
        min_agent_travel_dist = (self.env_bredth + self.env_width) * 0.125 # quarter of average boundary len
        start_buffer = 2.

        for agent_iterator in range(self.num_agents):
            # for each agent, attempt to find a start and goal that won't conflict with the starts, goals 
            # for previous agents
            new_start = None
            while True:
                # look for a start that is inside the environment
                new_start = agents[agent_iterator].get_start(self.env_width, self.env_bredth, env_start_end_buffer, loc_rng)
                found_conflict = False
                for j in range(len(starts)):
                    # make sure the new start isn't too close to an existing start
                    if self.euclidean_distance((starts[j][0], starts[j][1]), 
                                          (new_start[0], new_start[1])) < start_buffer:
                        found_conflict = True
                        break 
                
                # for j in range(len(goals)):
                #     # make sure the new start isn't too close to an existing goal
                #     if euclidean_distance((goals[j][0], goals[j][1]), 
                #                           (new_start[0], new_start[1])) < goal_buffer:
                #         found_conflict = True
                #         break 
                
                if not found_conflict:
                    break           

            starts.append(new_start)
            keep_looking_for_goal = True 
            while keep_looking_for_goal:
                # look for a goal that isn't too close to the agent's start or any other goal
                new_goal_radius = goal_radius
                new_goal_center = (loc_rng.uniform(env_start_end_buffer+new_goal_radius, self.env_width-env_start_end_buffer-new_goal_radius), 
                          loc_rng.uniform(env_start_end_buffer+new_goal_radius, self.env_bredth-env_start_end_buffer-new_goal_radius))

                # check if new start is too close to the new goal 
                if self.euclidean_distance(new_start, new_goal_center) < min_agent_travel_dist:
                    continue

                # make sure goal isn't too close to existing goal
                keep_looking_for_goal = False
                for goal, goal_radius in zip(goals, goal_radii):
                    if self.euclidean_distance(goal, new_goal_center) <= goal_radius + new_goal_radius + goal_buffer:
                        keep_looking_for_goal = True
                        break
                if not keep_looking_for_goal:
                    # if the goal is valid, save it off
                    goals.append(new_goal_center)
                    goal_radii.append(new_goal_radius)
                    goal_area += 2.*np.pi*new_goal_radius
            # end loop over goal gen 
        # end loop over start/goal gen  
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
         
        loc_rng = np.random.default_rng(seed)
        obs_radii = []
        obs_centers = []
        obs_objs = []

        # hardcoded parms, adjust if needed 
        obs_alley_buffer = 1.7 # min dist between obs 
        obs_start_buffer = 1.5 # min dist to agent starts
        obs_boundary_buffer = 0 # min dist to env boundary
        max_obs_radius = 4
        min_obs_radius = 1 # minimum obstacle dimension 
        failed_obs_creation_count = 0 # current number of failed random obs creation attempts, reset
                                      # on success
        obs_reset_count = 50000

        while (len(obs_centers) < self.num_obs):
            # While the area max isn't hit nor the failed obs creation count, 
            # attempt to generate a random obstacle
            found_valid_obs = True 
            new_obs = None
            new_obs_radius = 0
            new_obs_center = 0

            # get new obs center/radius
            new_obs_type = loc_rng.choice(self.obstacle_types)
            if new_obs_type is CircularObstacle2DTimed:
                new_obs_radius = loc_rng.uniform(min_obs_radius, max_obs_radius)
                new_obs_center = (loc_rng.uniform(obs_boundary_buffer+new_obs_radius, self.env_width-obs_boundary_buffer-new_obs_radius), 
                            loc_rng.uniform(obs_boundary_buffer+new_obs_radius, self.env_bredth-obs_boundary_buffer-new_obs_radius))
                new_obs = CircularObstacle2DTimed(new_obs_center[0], new_obs_center[1], new_obs_radius)
            elif new_obs_type is RectangleObstacle2D:
                new_obs_width = loc_rng.uniform(min_obs_radius, max_obs_radius)
                new_obs_height = loc_rng.uniform(min_obs_radius, max_obs_radius) 
                # This is a max approx used for overlap checks later
                new_obs_radius = 0.5 * np.sqrt(new_obs_width**2 + new_obs_height**2) 
                new_obs_center = (loc_rng.uniform(obs_boundary_buffer+new_obs_width, self.env_width-obs_boundary_buffer-new_obs_width), 
                            loc_rng.uniform(obs_boundary_buffer+new_obs_height, self.env_bredth-obs_boundary_buffer-new_obs_height))
                new_obs = RectangleObstacle2D(new_obs_center[0], new_obs_center[1], new_obs_width, new_obs_height)
            else:
                raise Exception("Some weird obs type found, this shouldn't happen!")
            
            # check against starts
            for start in starts:
                if new_obs.check_collision(start, 0, obstacle_buffer=obs_start_buffer):
                    found_valid_obs = False 
                    break

            # check against goals
            if found_valid_obs:
                for goal, goal_radius in zip(goals, goal_radii):
                    if new_obs.check_collision(goal, goal_radius+0.5):
                        found_valid_obs = False 
                        break
            
            # check against other obstacles 
            if found_valid_obs:
                for obs_center, obs_radius in zip(obs_centers, obs_radii):
                    if new_obs.check_collision(obs_center, obs_radius, obstacle_buffer=obs_alley_buffer):
                        found_valid_obs = False 
                        break

            # check if new obs valid 
            if found_valid_obs:
                failed_obs_creation_count = 0

                obs_centers.append(new_obs_center)
                obs_radii.append(new_obs_radius)
                obs_objs.append(new_obs)
            else: 
                failed_obs_creation_count +=1

            if failed_obs_creation_count > obs_reset_count:
                print("Reseting after generating", len(obs_objs), "obstacles.")
                seed *= 7
                starts, goals, goal_radii, goal_area = self.get_starts_goals(seed, agents)
                obs_radii = []
                obs_centers = []
                obs_objs = []

        print("Created env with " + str(len(obs_objs)) + " obstacles.")



        # end loop over obs creation
        return agents, starts, obs_objs, goals, goal_radii
    
from test_classes import *
from agent_builders import *

test_classes = [PrrtTestClass(collision_checks=False,printenv=False), KcbsTestClass()]
# collect agent builders
agent_builders = [SecondOrderCarBuilder()]
# instantiate test pipeline 
tp = TestPipelinePaperRandom(test_classes, agent_builders, test_rounds=1, num_agents=10, master_seed=100, num_obs=11, obstacle_types=[RectangleObstacle2D])
tp.run()

# from figure 3d of the KCBS paper on arxiv: https://arxiv.org/pdf/2207.00576
for num_agents in [20]:
    # collect test classes
    # test_classes = [KcbsTestClass(), PrrtTestClass(collision_checks=True)]
    test_classes = [PrrtTestClass(collision_checks=False,printenv=False), KcbsTestClass()]
    # collect agent builders
    agent_builders = [SecondOrderCarBuilder()]
    # instantiate test pipeline 
    tp = TestPipelinePaperRandom(test_classes, agent_builders, test_rounds=50, num_agents=num_agents, master_seed=num_agents*100, num_obs=11, obstacle_types=[RectangleObstacle2D])

    tp.run()
    tp.print_stats(filename="test_results/KCBS_Paper_33x33_Env_KCBS_PRRT_" + str(num_agents), plots=True)

# for num_agents in range(10):
#     # collect test classes
#     # test_classes = [KcbsTestClass(), PrrtTestClass(collision_checks=True)]
#     test_classes = [PrrtTestClass(collision_checks=True,printenv=False, max_planning_time=300), KcbsTestClass(printenv=False, max_planning_time=300), ]
#     # collect agent builders
#     agent_builders = [SecondOrderCarBuilder()]
#     # instantiate test pipeline 
#     tp = TestPipelinePaperRandom(test_classes, agent_builders, test_rounds=5, num_agents=num_agents, master_seed=315,
#                                  num_obs=9, obstacle_types=[RectangleObstacle2D], env_width=15, env_bredth=15)

#     tp.run()
#     tp.print_stats("Paper_15x15_OBS_PRRT_KCBS" + str(num_agents))

