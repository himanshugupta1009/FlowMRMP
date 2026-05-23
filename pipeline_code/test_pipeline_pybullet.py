import sys
sys.path.append('../src')
sys.path.append('../')
sys.path.append('.')

import importlib  
pybullet_utils = importlib.import_module("pybullet-planning.pybullet_tools.utils")
from Environments import *
from pybullet_env import PyBulletTurtle, PyBulletEnv, CircleObsPybullet, RectangleObsPybullet, AgentObsPybullet
import random
from utils import euclidean_distance
import numpy.ma as ma

class TestPipelinePB():
    def __init__(self, test_classes, test_rounds=100, num_agents=5, env_width=20, env_bredth=20, master_seed=42):
        """Initialize a new PB test environment

        Args:
            test_classes: List of test classes (see below for examples)
            test_rounds (int, optional): Number of test rounds to run, where a random env is 
                created and run through each of the test classes. Defaults to 100.
            num_agents (int, optional): Number of agents in each round. Defaults to 5.
            env_width (int, optional): Env 'x value.' Defaults to 20.
            env_bredth (int, optional): Env 'y value.' Defaults to 20.
            master_seed (int, optional): Test pipeline rng seed. Defaults to 42.
        """
        self.test_classes = test_classes
        self.test_rounds = test_rounds
        self.num_agents = num_agents
        self.agent_types = [PyBulletTurtle]
        self.env_width = env_width
        self.env_bredth = env_bredth
        self.rng = np.random.default_rng(master_seed)

    def get_agents(self, seed):
        """Gets agents for each pipeline round

        Args:
            seed (int): rng seed

        Returns:
            list(PybulletAgent): list of randomly-selected agents
        """
        agents = []
        for i in range(self.num_agents):
            agents.append(random.choice(self.agent_types)(
                10, # speed 
                agent_id=i, 
                rng_seed=seed+i 
            ))
        return agents
    
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
        # hardcoded param...for now
        goal_radius = 1.
        env_start_end_buffer = 1
        goal_area = 0.
        min_agent_travel_dist = (self.env_bredth + self.env_width) * 0.125 # quarter of average boundary len
        start_buffer = 2

        for i in range(self.num_agents):
            # for each agent, attempt to find a start and goal that won't conflict with the starts, goals 
            # for previous agents
            new_start = None
            while True:
                # look for a start that is inside the environment
                new_start = (loc_rng.uniform(env_start_end_buffer, self.env_width-env_start_end_buffer), 
                            loc_rng.uniform(env_start_end_buffer, self.env_bredth-env_start_end_buffer))
                found_conflict = False
                for i in range(len(starts)):
                    # make sure the new start isn't too close to an existing start
                    if euclidean_distance(agents[i].state_to_euclid(starts[i]), new_start) < start_buffer:
                        found_conflict = True
                        break 
                
                if not found_conflict:
                    break           

            starts.append(agents[i].get_start_from_euclid(new_start))
            keep_looking_for_goal = True 
            while keep_looking_for_goal:
                # look for a goal that isn't too close to the agent's start or any other goal
                new_goal_radius = goal_radius
                new_goal_center = (loc_rng.uniform(env_start_end_buffer+new_goal_radius, self.env_width-env_start_end_buffer-new_goal_radius), 
                          loc_rng.uniform(env_start_end_buffer+new_goal_radius, self.env_bredth-env_start_end_buffer-new_goal_radius))

                # check if new start is too close to the new goal 
                if euclidean_distance(new_start, new_goal_center) < min_agent_travel_dist:
                    continue

                # make sure goal isn't too close to existing goal
                keep_looking_for_goal = False
                for goal, goal_radius in zip(goals, goal_radii):
                    if euclidean_distance(goal, new_goal_center) <= goal_radius + new_goal_radius:
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

        Raises:
            Exception: when an invalid obstacle type is found

        Returns:
            agents, starts, obs_objs, goals, goal_radii lists for each agent
        """
        agents = self.get_agents(seed)
        starts, goals, goal_radii, goal_area = self.get_starts_goals(seed, agents)
         
        obs_area = 0. 
        loc_rng = np.random.default_rng(seed)
        obs_radii = []
        obs_centers = []
        obs_objs = []

        # hardcoded parms, adjust if needed 
        obs_alley_buffer = 0.75 # min dist between obs 
        obs_start_buffer = 0.4 # min dist to agent starts
        obs_boundary_buffer = 0.1 # min dist to env boundary
        max_obs_goal_area = 0.6 * (self.env_bredth * self.env_width) # target obstacle coverage percentage
        max_obs_radius = (self.env_bredth + self.env_width) * 0.125 # quarter of average boundary len
        min_obs_radius = 0.25 # minimum obstacle dimension 
        failed_obs_creation_attempts = 700 # number of failed random obs creation attempts before failure
        failed_obs_creation_count = 0 # current number of failed random obs creation attempts, reset
                                      # on success

        while ((goal_area + obs_area <= max_obs_goal_area) 
               and failed_obs_creation_count <= failed_obs_creation_attempts):
            # While the area max isn't hit nor the failed obs creation count, 
            # attempt to generate a random obstacle
            found_valid_obs = True 
            new_obs = None
            new_obs_radius = 0
            new_obs_center = 0

            # get new obs center/radius
            new_obs_type = random.choice([CircleObsPybullet, RectangleObsPybullet])
            if new_obs_type is CircleObsPybullet:
                new_obs_radius = loc_rng.uniform(min_obs_radius, max_obs_radius)
                new_obs_center = (loc_rng.uniform(obs_boundary_buffer+new_obs_radius, self.env_width-obs_boundary_buffer-new_obs_radius), 
                            loc_rng.uniform(obs_boundary_buffer+new_obs_radius, self.env_bredth-obs_boundary_buffer-new_obs_radius))
                new_obs = new_obs_type(new_obs_center[0], new_obs_center[1], new_obs_radius)
            elif new_obs_type is RectangleObsPybullet:
                new_obs_len = loc_rng.uniform(min_obs_radius, max_obs_radius)
                new_obs_width = loc_rng.uniform(min_obs_radius, max_obs_radius) 
                # This is a max approx used for overlap checks later
                new_obs_radius = 0.5 * np.sqrt(new_obs_len**2 + new_obs_width**2) 
                new_obs_center = (loc_rng.uniform(obs_boundary_buffer+new_obs_width, self.env_width-obs_boundary_buffer-new_obs_width), 
                            loc_rng.uniform(obs_boundary_buffer+new_obs_len, self.env_bredth-obs_boundary_buffer-new_obs_len))
                new_obs = new_obs_type(new_obs_center[0], new_obs_center[1], new_obs_len, new_obs_width)
            else:
                raise Exception("Some weird obs type found, this shouldn't happen!")
            
            # check against starts
            for start, agent in zip(starts, agents):
                if euclidean_distance(agent.state_to_euclid(start), new_obs_center) < new_obs_radius + obs_start_buffer:
                    found_valid_obs = False 
                    break

            # check against goals
            if found_valid_obs:
                for goal, goal_radius in zip(goals, goal_radii):
                    if euclidean_distance(goal, new_obs_center) < new_obs_radius + goal_radius:
                        found_valid_obs = False 
                        break
            
            # check against other obstacles 
            if found_valid_obs:
                for obs_center, obs_radius in zip(obs_centers, obs_radii):
                    if euclidean_distance(obs_center, new_obs_center) < new_obs_radius + obs_radius + obs_alley_buffer:
                        found_valid_obs = False 
                        break

            # check if new obs valid 
            if found_valid_obs:
                failed_obs_creation_count = 0

                obs_centers.append(new_obs_center)
                obs_radii.append(new_obs_radius)
                obs_objs.append(new_obs)
                if new_obs_type is RectangleObsPybullet:
                    obs_area += new_obs.l * new_obs.w 
                elif new_obs_type is CircleObsPybullet:
                    obs_area += 2.*np.pi*new_obs.r 
                else:
                    raise Exception("Some weird obs type found, this shouldn't happen!")
            else: 
                failed_obs_creation_count +=1
        # end loop over obs creation 
        return agents, starts, obs_objs, goals, goal_radii
    
    def run(self):
        """
        Runs self.test_rounds environments through each test class
        """
        for seed in range(100,100+self.test_rounds):
            agents, starts, obstacles, goals, goal_radii = self.get_env_parms(seed)
            for test_class in self.test_classes:
                (passed, dt, costs) = test_class.test_func(agents, starts, obstacles, goals, goal_radii, 
                                                           self.env_width, self.env_bredth, seed)
                test_class.success.append(passed)
                test_class.times.append(dt)
                test_class.costs.append(sum(costs)) 
    
    def print_stats(self, filename=None):
        """
        Displays stats from the test run 

        Args:
            filename (str, optional): File to save results to if not None. 
                If None, will print to stdout. Defaults to None.
        """
        output = "Num agents: " + str(self.num_agents) + "\n\n"

        for test_class in self.test_classes:
            times = np.asarray(test_class.times)
            costs = np.asarray(test_class.costs)
            success = np.asarray(test_class.success)

            successes = np.count_nonzero(success)
            sum_total_cost = np.sum(ma.masked_array(costs, ~success))
            std_costs = np.std(ma.masked_array(costs, ~success), ddof=1)
            sem_costs = std_costs / np.sqrt(successes)
            sum_time = np.sum(ma.masked_array(times, ~success))
            std_time = np.std(ma.masked_array(times, ~success), ddof=1)
            sem_time = std_time / np.sqrt(successes)
            

            output += (test_class.name + ":" 
                + "\n\tPercent Success: " + str(successes/self.test_rounds) 
                + "\n\tAverage Total Cost: " + str(sum_total_cost/successes)
                + "\n\tSTD Total Cost: " + str(std_costs)
                + "\n\tSEM Total Cost: " + str(sem_costs)
                + "\n\tAverage Time: " + str(sum_time/successes) 
                + "\n\tSTD Time: " + str(std_time)
                + "\n\tSEM Time: " + str(sem_time)
                + "\n\n")

        if filename == None:
            print(output)
        else:
            with open(filename + '_agents' + str(self.num_agents) + '.txt', 'w+') as f:
                f.write(output)

"""
PRRT EB Test Class
"""
from edge_bundle import EdgeBundle
from pybullet_env import *
from prrt_eb import EdgeBundlePRRT
import math
import time 

edge_bundle_file_location = 'edge_bundles/eb_pb_turtle_speed_10_edges-10000.npz' 
data = np.load(edge_bundle_file_location, allow_pickle=True)
eb_pbt = EdgeBundle(data, fix_num_edges=1000)

class PrrtEbTestClass:
    def __init__(self):
        self.name = "PRRT With Edge Bundles"
        self.costs = []
        self.times = []
        self.success = []

    def test_func(self, agents, starts, obstacles, goals, goal_radii, env_width, env_bredth, seed, use_gui = True):
        env = PyBulletEnv(env_width, env_bredth, obstacles, use_gui=use_gui, speed=20)

        isvalids = [agents[i].is_new_node_valid for i in range(len(agents))]
        gcost = [agents[i].get_cost for i in range(len(agents))]
        rpfs = [agents[i].get_random_point for i in range(len(agents))]
        arg = [agents[i].agent_reached_goal for i in range(len(agents))]
        tfs = [agents[i].point_translate_function for i in range(len(agents))]
        # TODO: Get this from an agent builder or something
        ebs = [eb_pbt for _ in range(len(agents))]

        start_time = time.time()
        (paths, states, rrts, controls, timesteps) = EdgeBundlePRRT.plan_multi(agents=agents, 
                                            starts=starts,
                                            goals=goals,
                                            goal_radii=goal_radii,
                                            env=env,
                                            edge_bundle=ebs,
                                            # change these after debug 
                                            max_iter = 15000, planning_time=180,   
                                            isvalid_function=isvalids, 
                                            cost_function=gcost,
                                            random_point_function=rpfs, 
                                            reached_goal_function = arg,
                                            translate_function=tfs,
                                            udf_seed = seed,
                                            obs_type=AgentObsPybullet
                                            )
        total_time = time.time() - start_time
        if use_gui:
            pybullet_utils.disconnect()

        for rrt in rrts:
            if rrt.path_found == False:
                return (False, 0, []) 

        costs = []
        for (path, rrt) in zip(paths, rrts):
            rrtNode = rrt.tree.nodes(data=True)[path[-1]].get('value')
            costs.append(rrtNode.cost_so_far)

        return (True, total_time, costs)
    
"""
PRRT Test Class
"""
from prrt import PRRT

class PrrtTestClass:
    def __init__(self):
        self.name = "PRRT"
        self.costs = []
        self.times = []
        self.success = []

    def test_func(self, agents, starts, obstacles, goals, goal_radii, env_width, env_bredth, seed, use_gui = True):
        env = PyBulletEnv(env_width, env_bredth, obstacles, use_gui=use_gui, speed=20)

        isvalids = [agents[i].is_new_node_valid for i in range(len(agents))]
        gcost = [agents[i].get_cost for i in range(len(agents))]
        rpfs = [agents[i].get_random_point for i in range(len(agents))]
        arg = [agents[i].agent_reached_goal for i in range(len(agents))]

        start_time = time.time()
        (paths, states, controls, timesteps, rrts) = PRRT.plan_multi(agents=agents, 
                                            starts=starts,
                                            goals=goals,
                                            goal_radii=goal_radii,
                                            env=env,
                                            # change these after debug 
                                            max_iter = 15000, planning_time=180,   
                                            # If we want different agent types, need to change this!      
                                            isvalid_function=isvalids,
                                            cost_function=gcost,
                                            random_point_function=rpfs, 
                                            reached_goal_function = arg,
                                            udf_seed = seed,
                                            obs_type=AgentObsPybullet,
                                            use_fixed_sampling_time=False, 
                                            sampling_time_step=2.0
                                            )
        total_time = time.time() - start_time
        if use_gui:
            pybullet_utils.disconnect()

        for rrt in rrts:
            if rrt.path_found == False:
                return (False, 0, []) 

        costs = []
        for (path, rrt) in zip(paths, rrts):
            rrtNode = rrt.tree.nodes(data=True)[path[-1]].get('value')
            costs.append(rrtNode.cost_so_far)


        return (True, total_time, costs)
    
# TODO: KCBS classes, at least one other class of agent
    
test_classes = [PrrtTestClass(), PrrtEbTestClass()]
tp = TestPipelinePB(test_classes, test_rounds=2, num_agents=3, master_seed=40)

tp.run()
tp.print_stats()