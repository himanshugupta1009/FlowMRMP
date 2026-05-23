from contextlib import redirect_stdout
import os
import random 
import numpy as np
import numpy.ma as ma
import pandas as pd
import traceback
import copy

os.environ.setdefault("MPLCONFIGDIR", "/tmp/mapf_matplotlib_cache")
os.makedirs(os.environ["MPLCONFIGDIR"], exist_ok=True)
import matplotlib

matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import seaborn as sns
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed

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

class TestPipeline():
    def __init__(self, test_classes, agent_builders, test_rounds=100, num_agents=5, env_width=40, env_bredth=40, env_height=None, 
                 master_seed=42, goal_radius = 0.5, 
                 savepath="", obstacle_classes = [CircularObstacle2D], processes = 1):
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
        """
        self.test_classes = test_classes
        self.test_rounds = test_rounds
        self.num_agents = num_agents
        self.agent_builders = agent_builders
        self.obstacle_classes = obstacle_classes
        for obstacle_class in obstacle_classes:
            if not (obstacle_class is CircularObstacle2D or obstacle_class is RectangleObstacle2D 
                    or obstacle_class is CuboidObstacle3D):
                raise Exception("Unsupported obstacle class: " + str(obstacle_class))
        self.env_width = env_width
        self.env_bredth = env_bredth
        self.env_height = env_height
        self.rng = np.random.default_rng(master_seed)
        self.master_seed = master_seed
        self.goal_radius = goal_radius
        self.savepath = savepath
        self.processes = processes

    def euclidean_distance(self, p1, p2): #p1 and p2 are tuples
        # return np.linalg.norm(np.array(p1) - np.array(p2))
        d = np.sqrt(sum((a - b) ** 2 for a, b in zip(p1, p2)))
        return d

    def get_agents(self, seed):
        """
        Gets agents for each pipeline round

        Args:
            seed (int): rng seed

        Returns:
            list(AgentBuilder): List of agent builders randomly selected
        """
        loc_rng = np.random.default_rng(seed)
        agents = []
        for i in range(self.num_agents):
            builder = copy.deepcopy(loc_rng.choice(self.agent_builders))
            builder.initialize_identification(i, seed+i)
            agents.append(builder)
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
        # hardcoded params...for now
        goal_radius = 2.
        env_start_end_buffer = 3.
        goal_area = 0.
        min_agent_travel_dist = (self.env_bredth + self.env_width) * 0.125 # quarter of average boundary len
        start_buffer = 3.

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
                    if self.euclidean_distance(goal, new_goal_center) <= goal_radius + new_goal_radius:
                        keep_looking_for_goal = True
                        break 
                if not keep_looking_for_goal:
                    # if the goal is valid, save it off
                    goals.append(new_goal_center)
                    goal_radii.append(new_goal_radius)
                    goal_area += np.pi*new_goal_radius*new_goal_radius
            # end loop over goal gen 
        # end loop over start/goal gen  
        return starts, goals, goal_radii, goal_area

    def circle_circle_collision(self, cir1_center, cir1_radius, cir2_center, cir2_radius, buffer):
        """Check if a circle is within a buffer to a circle

        Args:
            cir1_center (tuple[float, float]): Center of the first circle
            cir1_radius (float): Radius of the first circle
            cir2_center (tuple[float, float]): Center of the second circle
            cir2_radius (float): Radius of the second circle
            buffer (float): Buffer distance between circles

        Returns:
            boolean: True if the circles are within the buffer distance; False otherwise
        """
        center_dist = self.euclidean_distance(cir1_center, cir2_center)
        return center_dist - (cir1_radius + cir2_radius) < buffer
    

    def rectangle_circle_collision(self, 
        rect_center, rect_width, rect_height,
        circle_center, circle_radius,
        buffer_distance
    ):
        """
        Determine whether an axis-aligned rectangle and a circle are closer at any
        point than a specified buffer distance.

        Parameters
        ----------
        rect_center : tuple[float, float]
            (x, y) center of the rectangle.
        rect_width : float
            Width of the rectangle.
        rect_height : float
            Height of the rectangle.
        circle_center : tuple[float, float]
            (x, y) center of the circle.
        circle_radius : float
            Radius of the circle.
        buffer_distance : float
            Minimum allowed separation distance.

        Returns
        -------
        bool
            True if the rectangle and circle are closer than buffer_distance
            at any point; False otherwise.
        """

        rx, ry = rect_center
        cx, cy = circle_center

        # Rectangle half-dimensions
        half_w = rect_width / 2.0
        half_h = rect_height / 2.0

        # Rectangle bounds
        min_x = rx - half_w
        max_x = rx + half_w
        min_y = ry - half_h
        max_y = ry + half_h

        # Clamp circle center to rectangle to find closest point
        closest_x = min(max(cx, min_x), max_x)
        closest_y = min(max(cy, min_y), max_y)

        # Distance from circle center to closest point on rectangle
        dx = cx - closest_x
        dy = cy - closest_y
        distance_center_to_rect = math.hypot(dx, dy)

        # Minimum distance between shapes
        min_distance = distance_center_to_rect - circle_radius

        return min_distance < buffer_distance
    
    def rectangle_rectangle_collision( self, 
        rect1_center, rect1_width, rect1_height,
        rect2_center, rect2_width, rect2_height,
        buffer_distance
    ):
        """
        Determine whether two axis-aligned rectangles are closer at any point
        than a specified buffer distance.

        Parameters
        ----------
        rect1_center : tuple[float, float]
            (x, y) center of rectangle 1.
        rect1_width : float
            Width of rectangle 1.
        rect1_height : float
            Height of rectangle 1.
        rect2_center : tuple[float, float]
            (x, y) center of rectangle 2.
        rect2_width : float
            Width of rectangle 2.
        rect2_height : float
            Height of rectangle 2.
        buffer_distance : float
            Minimum allowed separation distance.

        Returns
        -------
        bool
            True if the rectangles are closer than buffer_distance at any point;
            False otherwise.
        """

        x1, y1 = rect1_center
        x2, y2 = rect2_center

        # Half-dimensions
        hw1, hh1 = rect1_width / 2.0, rect1_height / 2.0
        hw2, hh2 = rect2_width / 2.0, rect2_height / 2.0

        # Separation along each axis
        dx = abs(x1 - x2) - (hw1 + hw2)
        dy = abs(y1 - y2) - (hh1 + hh2)

        # If rectangles overlap along an axis, separation is zero on that axis
        dx = max(dx, 0.0)
        dy = max(dy, 0.0)

        # Minimum distance between rectangles
        min_distance = math.hypot(dx, dy)

        return min_distance < buffer_distance
    
    def obs_obs_collision(self, obs1, obs2, buffer):
        if isinstance(obs1, CircularObstacle2D) and isinstance(obs2, CircularObstacle2D):
            return self.circle_circle_collision(
                (obs1.x, obs1.y), obs1.r,
                (obs2.x, obs2.y), obs2.r,
                buffer
            )
        elif isinstance(obs1, RectangleObstacle2D) and isinstance(obs2, CircularObstacle2D):
            return self.rectangle_circle_collision(
                (obs1.x, obs1.y), obs1.w, obs1.h,
                (obs2.x, obs2.y), obs2.r,
                buffer
            )
        elif isinstance(obs1, CircularObstacle2D) and isinstance(obs2, RectangleObstacle2D):
            return self.rectangle_circle_collision(
                (obs2.x, obs2.y), obs2.w, obs2.h,
                (obs1.x, obs1.y), obs1.r,
                buffer
            )
        elif isinstance(obs1, RectangleObstacle2D) and isinstance(obs2, RectangleObstacle2D):
            return self.rectangle_rectangle_collision(
                (obs1.x, obs1.y), obs1.w, obs1.h,
                (obs2.x, obs2.y), obs2.w, obs2.h,
                buffer
            )
        else:
            raise Exception("Unsupported obstacle types for collision checking: " + str(type(obs1)) + ", " + str(type(obs2)))
        
    def obs_circle_collision(self, obs, circle_center, circle_radius, buffer):
        if isinstance(obs, CircularObstacle2D):
            return self.circle_circle_collision(
                (obs.x, obs.y), obs.r,
                circle_center, circle_radius,
                buffer
            )
        elif isinstance(obs, RectangleObstacle2D):
            return self.rectangle_circle_collision(
                (obs.x, obs.y), obs.w, obs.h,
                circle_center, circle_radius,
                buffer
            )
        else:
            raise Exception("Unsupported obstacle type for collision checking: " + str(type(obs)))

    
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
         
        obs_area = 0. 
        loc_rng = np.random.default_rng(seed)
        obs_objs = []

        # hardcoded parms, adjust if needed 
        obs_alley_buffer = 2.7 # min dist between obs 
        obs_start_buffer = 2.7 # min dist to agent starts
        obs_boundary_buffer = 2.7 # min dist to env boundary
        obs_goal_buffer = 0.75
        max_obs_goal_area = 0.2 * (self.env_bredth * self.env_width) # target obstacle coverage percentage
        max_obs_radius = (self.env_bredth + self.env_width) * 0.125 # quarter of average boundary len
        min_obs_radius = 0.5 # minimum obstacle dimension 
        failed_obs_creation_attempts = 1700 # number of failed random obs creation attempts before failure
        failed_obs_creation_count = 0 # current number of failed random obs creation attempts, reset
                                      # on success

        while ((goal_area + obs_area <= max_obs_goal_area) 
               and failed_obs_creation_count <= failed_obs_creation_attempts):
            # While the area max isn't hit nor the failed obs creation count, 
            # attempt to generate a random obstacle
            found_valid_obs = True 
            new_obs = None
            new_obs_area = 0

            # get new obs center/radius
            new_obs_type = loc_rng.choice(self.obstacle_classes)
            if new_obs_type is CircularObstacle2D:
                new_obs_radius = loc_rng.uniform(min_obs_radius, max_obs_radius)
                new_obs_center = (loc_rng.uniform(obs_boundary_buffer+new_obs_radius, self.env_width-obs_boundary_buffer-new_obs_radius), 
                            loc_rng.uniform(obs_boundary_buffer+new_obs_radius, self.env_bredth-obs_boundary_buffer-new_obs_radius))
                new_obs = CircularObstacle2D(new_obs_center[0], new_obs_center[1], new_obs_radius)
                new_obs_area = np.pi * new_obs_radius * new_obs_radius
            elif new_obs_type is RectangleObstacle2D:
                new_obs_width = loc_rng.uniform(min_obs_radius*2., max_obs_radius*2.)
                new_obs_bredth = loc_rng.uniform(min_obs_radius*2., max_obs_radius*2.)
                new_obs_center = (loc_rng.uniform(obs_boundary_buffer+new_obs_width/2., self.env_width-obs_boundary_buffer-new_obs_width/2.), 
                            loc_rng.uniform(obs_boundary_buffer+new_obs_bredth/2., self.env_bredth-obs_boundary_buffer-new_obs_bredth/2.))
                new_obs = RectangleObstacle2D(new_obs_center[0], new_obs_center[1], new_obs_width, new_obs_bredth)
                new_obs_area = new_obs_width * new_obs_bredth   
            else:
                raise Exception("Unsupported obstacle class: " + str(new_obs_type))

            # check against starts
            for start in starts:
                if self.obs_circle_collision(new_obs, (start[0], start[1]), 0., obs_start_buffer):
                    found_valid_obs = False 
                    break

            # check against goals
            if found_valid_obs:
                for goal, goal_radius in zip(goals, goal_radii):
                    if self.obs_circle_collision(new_obs, (goal[0], goal[1]), goal_radius, obs_goal_buffer):
                        found_valid_obs = False
                        break
            
            # check against other obstacles 
            if found_valid_obs:
                for old_obs_obj in obs_objs:
                    if self.obs_obs_collision(new_obs, old_obs_obj, obs_alley_buffer):
                        found_valid_obs = False
                        break

            # check if new obs valid 
            if found_valid_obs:
                failed_obs_creation_count = 0

                obs_objs.append(new_obs)
                obs_area += new_obs_area
            else: 
                failed_obs_creation_count +=1

        print("Created env with " + str(len(obs_objs)) + " obstacles covering " + str(obs_area/(self.env_bredth * self.env_width)) + " of the env.")

        if (self.savepath != ""):
            filedir = self.savepath + "/env_checks"
            os.makedirs(filedir, exist_ok=True) 
            agent_objs = []
            for agent in agents:
                agent_objs.append(agent.get_agent())
            pcol = ['xkcd:powder pink', 'xkcd:metallic blue', 'xkcd:pastel orange', 'xkcd:pastel blue', 'xkcd:terracotta', 'xkcd:purplish grey',
                        'xkcd:powder pink', 'xkcd:metallic blue', 'xkcd:pastel orange', 'xkcd:pastel blue', 'xkcd:terracotta', 'xkcd:purplish grey']*3
            env = SquareEnvironment(self.env_width, self.env_bredth, obs_objs, obs_buffers=False)
            MultiRRTPrinter.print_rrt_env(filedir + '/check_' + str(seed) + '.png',
                                                env, agent_objs, starts, goals, [self.goal_radius for _ in starts], pcol)

        # end loop over obs creation 
        return agents, starts, obs_objs, goals, [self.goal_radius for _ in starts] #goal_radii
    
    def _run_seed_worker(self, 
                         seed,
                         test_classes,
                         env_width,
                         env_bredth,
                         env_height,
                         get_env_parms):
        """
        Runs all test classes for a single seed.
        Returns structured results for parent aggregation.
        """

        print("Running seed " + str(seed))

        results = {}

        agents, starts, obstacles, goals, goal_radii = get_env_parms(seed)

        for test_class in test_classes:
            cls_name = test_class.short_name

            try:
                passed, dt, cost, path_times, max_time, message = test_class.test_func(
                    agents,
                    starts.copy(),
                    obstacles.copy(),
                    goals.copy(),
                    goal_radii.copy(),
                    seed,
                    env_width,
                    env_bredth, 
                    env_height
                )

                results[cls_name] = {
                    "success": passed,
                    "time": dt,
                    "cost": cost,
                    "path_times": path_times,
                    "max_time": max_time,
                    "exception": None,
                    "message": message
                }

            except Exception as e:
                results[cls_name] = {
                    "success": False,
                    "time": None,
                    "cost": None,
                    "path_times": None,
                    "max_time": None,
                    "exception": f"For round {seed}: {e}\n{traceback.format_exc()}",
                    "message": "Exception: see logs"
                }
                print(f'Exception for {cls_name} for round {seed}: {e}\n{traceback.format_exc()}', file=sys.stderr)

        return seed, results

    def _init_runner_process(self, test_classes, env_width, env_bredth, env_height, get_env_parms):
        """
        Warm up numba for each new process
        """
        
        print("Warming up test classes")
        agents, starts, obstacles, goals, goal_radii = get_env_parms(555)
        for test_class in test_classes:
            test_class.warmup_test(agents.copy(),
                    starts.copy(),
                    obstacles.copy(),
                    goals.copy(),
                    goal_radii.copy(),
                    555, # seed
                    env_width,
                    env_bredth,
                    env_height
                ) 
    
    def run(self):
        """
        Runs self.test_rounds environments through each test class,
        parallelized over seeds, with progress bar.
        """
        start = self.master_seed
        seeds = list(range(start, start + self.test_rounds))

        max_workers = self.processes

        with ProcessPoolExecutor(max_workers=max_workers, initializer=self._init_runner_process, initargs=(self.test_classes, self.env_width, self.env_bredth, self.env_height, self.get_env_parms)) as executor:
            futures = [
                executor.submit(
                    self._run_seed_worker,
                    seed,
                    self.test_classes,
                    self.env_width,
                    self.env_bredth,
                    self.env_height,
                    self.get_env_parms,
                )
                for seed in seeds
            ]

            # define a progress bar
            with tqdm(total=len(futures), desc="Running seeds", unit="seed") as pbar:
                for future in as_completed(futures):
                    seed, seed_results = future.result()

                    for test_class in self.test_classes:
                        cls_name = test_class.short_name
                        result = seed_results[cls_name]

                        if result["exception"] is None:
                            test_class.success[seed] = result["success"]
                            test_class.times[seed] = result["time"]
                            test_class.costs[seed] = result["cost"]
                            test_class.path_times[seed] = result["path_times"]
                            test_class.max_times[seed] = result["max_time"]
                            test_class.messages[seed] = result["message"]
                        else:
                            print("Exception:\n" + result["exception"])
                            test_class.exceptions.setdefault(seed, []).append(
                                result["exception"]
                            )
                            test_class.success[seed] = result["success"]
                            test_class.times[seed] = result["time"]
                            test_class.costs[seed] = result["cost"]
                            test_class.path_times[seed] = result["path_times"]
                            test_class.max_times[seed] = result["max_time"]
                            test_class.messages[seed] = result["message"]

                    pbar.update(1)

    
    def print_stats(self, print_to_console = True, filename=None, plots = False):
        """
        Displays stats from the test run 

        Args:
            print_to_console (bool, optional): Show resuls in stdout. 
                Defaults to True.
            filename (str, optional): File to save results to if not None. 
                Will always print to stdout. Defaults to None.
            plots (bool, optional): generate boxenplots. Defaults to False.
        """
        output = "Num agents: " + str(self.num_agents) + "\n\n"

        if filename:
            os.makedirs(filename, exist_ok=True) 

        # Collect all data across test classes
        all_times = []
        all_costs = []
        all_path_times = []
        all_max_times = []
        all_labels = []
        success_rate_labels = []
        success_rates = [] 
        seeds = np.asarray([i for i in range(self.master_seed, self.master_seed + self.test_rounds)])

        all_failed = []

        for test_class in self.test_classes:
            output += (test_class.name + ":\n")
            if len(test_class.success) > 0:
                times = np.asarray(list(dict(sorted(test_class.times.items())).values()))
                costs = np.asarray(list(dict(sorted(test_class.costs.items())).values()))
                success = np.asarray(list(dict(sorted(test_class.success.items())).values()), dtype=bool)
                path_time = np.asarray(list(dict(sorted(test_class.path_times.items())).values()))
                max_time = np.asarray(list(dict(sorted(test_class.max_times.items())).values()))
                messages = list(dict(sorted(test_class.messages.items())).values())
                # save to csv
                if filename:
                    try: 
                        os.makedirs(filename+'/csvs', exist_ok=True)
                        df = pd.DataFrame({
                            'Seed': seeds,
                            'Success': success,
                            'Computation Time (s)': times,
                            'Total Path Costs': costs,
                            'Average Agent Path Time (s)': path_time,
                            'Max Agent Path Time (s)': max_time,
                            'Messages': messages
                        })
                        df.to_csv(f"{filename}/csvs/{test_class.short_name.replace(' ', '_')}_results.csv", index=False)
                    except Exception as e:  
                        print("Failed to save csv for " + test_class.name + ": " + str(e))

                successes = np.count_nonzero(success)
                total_runs = len(success)
                success_rate_labels.append(test_class.short_name)
                success_rates.append((successes / total_runs) * 100.)
                
                if successes > 0:
                    successful_costs = ma.masked_array(costs, ~success)
                    successful_times = ma.masked_array(times, ~success)
                    successful_path_times = ma.masked_array(path_time, ~success)
                    successful_max_times = ma.masked_array(max_time, ~success)

                    sum_total_cost = np.sum(successful_costs)
                    std_costs = np.std(successful_costs, ddof=1)
                    sem_costs = std_costs / np.sqrt(successes)

                    sum_time = np.sum(successful_times)
                    std_time = np.std(successful_times, ddof=1)
                    sem_time = std_time / np.sqrt(successes)

                    sum_avg_path = np.sum(successful_path_times)
                    std_avg_path = np.std(successful_path_times, ddof=1)
                    sem_avg_path = std_avg_path / np.sqrt(successes)
                    
                    sum_max_path = np.sum(successful_max_times)
                    std_max_path = np.std(successful_max_times, ddof=1)
                    sem_max_path = std_max_path / np.sqrt(successes)

                    # Append only successful runs to combined metric lists.
                    all_times.extend(times[success])
                    all_costs.extend(costs[success])
                    all_path_times.extend(path_time[success])
                    all_max_times.extend(max_time[success])
                    all_labels.extend([test_class.short_name] * successes)

                    output += (
                            "\tTotal rounds completed: " + str(total_runs) 
                            + "\n\tPercent Success: " + str(successes/total_runs) 
                            + "\n\tAverage Total Cost: " + str(sum_total_cost/successes)
                            + "\n\tSTD Total Cost: " + str(std_costs)
                            + "\n\tSEM Total Cost: " + str(sem_costs)
                            + "\n\tAverage Time: " + str(sum_time/successes) 
                            + "\n\tSTD Time: " + str(std_time)
                            + "\n\tSEM Time: " + str(sem_time)
                            + "\n\tAverage Avg. Path Time: " + str(sum_avg_path/successes) 
                            + "\n\tSTD Avg. Path Time: " + str(std_avg_path)
                            + "\n\tSEM Avg. Path Time: " + str(sem_avg_path)
                            + "\n\tAverage Max Path Time: " + str(sum_max_path/successes) 
                            + "\n\tSTD Max Path Time: " + str(std_max_path)
                            + "\n\tSEM Max Path Time: " + str(sem_max_path)
                            + "\n\tMax Planning Time: " + str(test_class.max_planning_time)
                            + "\n")
                else:
                    all_failed.append(test_class.name)
                    output += (
                            "\tTotal rounds completed: " + str(total_runs) 
                            + "\n\tPercent Success: 0.0"
                            + "\n\tAverage Total Cost: N/A"
                            + "\n\tSTD Total Cost: N/A"
                            + "\n\tSEM Total Cost: N/A"
                            + "\n\tAverage Time: N/A"
                            + "\n\tSTD Time: N/A"
                            + "\n\tSEM Time: N/A"
                            + "\n\tAverage Avg. Path Time: N/A"
                            + "\n\tSTD Avg. Path Time: N/A"
                            + "\n\tSEM Avg. Path Time: N/A"
                            + "\n\tAverage Max Path Time: N/A"
                            + "\n\tSTD Max Path Time: N/A"
                            + "\n\tSEM Max Path Time: N/A"
                            + "\n\tMax Planning Time: " + str(test_class.max_planning_time)
                            + "\n")
            else:
                all_failed.append(test_class.name)
                output += (
                        "\tTotal rounds completed: 0"
                        + "\n\tPercent Success: N/A"
                        + "\n\tAverage Total Cost: N/A"
                        + "\n\tSTD Total Cost: N/A"
                        + "\n\tSEM Total Cost: N/A"
                        + "\n\tAverage Time: N/A"
                        + "\n\tSTD Time: N/A"
                        + "\n\tSEM Time: N/A"
                        + "\n\tAverage Avg. Path Time: N/A"
                        + "\n\tSTD Avg. Path Time: N/A"
                        + "\n\tSEM Avg. Path Time: N/A"
                        + "\n\tAverage Max Path Time: N/A"
                        + "\n\tSTD Max Path Time: N/A"
                        + "\n\tSEM Max Path Time: N/A"
                        + "\n\tMax Planning Time: " + str(test_class.max_planning_time)
                        + "\n")
            # end condition on success           

            output += "Exceptions:\n"
            for exception_message in test_class.exceptions:
                output += exception_message + "\n"

            output += "\n\n"
        # end loop over classes 

        metrics = {
            "Computation Time (s)": all_times,
            "Total Path Costs": all_costs,
            "Average Agent Path Time (s)": all_path_times,
            "Max Agent Path Time (s)": all_max_times
        }

        if print_to_console:
            print(output)

        if filename:
            with open(filename + '/stats.txt', "w") as f:
                f.write(output)

        if plots:
            unique_classes = list(dict.fromkeys(all_labels))  # preserve order
            palette = sns.color_palette("tab10", n_colors=len(unique_classes))
            palette_dict = {cls: palette[i] for i, cls in enumerate(unique_classes)}
            unique_success_rate_classes = list(dict.fromkeys(success_rate_labels))
            success_palette = sns.color_palette("tab10", n_colors=len(unique_success_rate_classes))
            success_palette_dict = {cls: success_palette[i] for i, cls in enumerate(unique_success_rate_classes)}

            plotsfile = "" if not filename else filename
            plotsfile = plotsfile + "/"

            # make boxenplots for each metric
            if len(all_labels) > 0:
                for metric_name, metric_data in metrics.items():
                    plt.figure(figsize=(10, 6))
                    sns.boxenplot(x=all_labels, y=metric_data, palette=palette_dict, hue=all_labels, showfliers=False)
                    plt.title(f"{metric_name} (Successful Runs Only) for " + str(self.num_agents) + " Agents")
                    plt.xlabel("Test Class")
                    plt.ylabel(metric_name)
                    plt.grid(True, alpha=0.3)
                    plt.tight_layout()

                    plot_filename = f"{plotsfile}{metric_name.replace(' ', '_')}_boxen.png"
                    plt.savefig(plot_filename)
                    plt.close()
            
            # make boxplot for success rate
            if len(success_rate_labels) > 0:
                plt.figure(figsize=(10, 6))
                sns.barplot(x=success_rate_labels, y=success_rates, palette=success_palette_dict, hue=success_rate_labels)
                plt.title("Success Rate per Test Class for " + str(self.num_agents) + " Agents")
                plt.xlabel("Test Class")
                plt.ylabel("Success Rate (%)")
                plt.ylim(0, 100)  # percentage scale
                plt.grid(True, alpha=0.3)
                plt.tight_layout()

                plot_filename = f"{plotsfile}successes_barplot.png"
                plt.savefig(plot_filename)
                plt.close()
            plt.close('all')

        return all_failed



from test_classes import *
from agent_builders import *

if __name__ == "__main__":
    # # collect agent builders
    agent_builders = [UnicycleBuilder(max_speed=0.5, max_omega=2.0, radius=0.4, \
                                       edge_bundle_file_location='edge_bundles/eb_unicycle_edges-1000_v05_av2.npz',
                                       kino_ti_edge_bundle_file_location='edge_bundles/eb_unicycle_dbCBS_kinodynamic_TI_edges_100000.npz')]
    num_processes = 10

    planning_time = 300.0
    for num_agents in [25, 30]:
        for gr in [1.0]:
            savepath = "test_results/final_test_results/RANDOM_ALL_UCYCLE_a" + str(num_agents) + "_gr" + str(gr)
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

            # test_classes = [HimanshuEnvTranslator(save_location=savepath)]

            # instantiate test pipeline 
            tp = TestPipeline(test_classes, agent_builders, test_rounds=100, num_agents=num_agents, master_seed=num_agents*100, goal_radius=gr,
                              savepath=savepath, obstacle_classes=[CircularObstacle2D, RectangleObstacle2D], processes=num_processes)
            with open(savepath+'/log.txt', 'w') as f, redirect_stdout(f):
                tp.run()
            tp.print_stats(filename=savepath, plots=True)

# for num_agents in range(2, 7):
#     # collect test classes
#     test_classes = [KcbsEbTestClass(), KcbsTestClass(), PrrtEbTestClass(collision_checks=True), PrrtTestClass(collision_checks=True)]
#     # collect agent builders
#     agent_builders = [UnicycleBuilder, MecanumBuilder]
#     # instantiate test pipeline 
#     tp = TestPipeline(test_classes, agent_builders, test_rounds=100, num_agents=num_agents, master_seed=num_agents*100)

#     tp.run()
#     tp.print_stats("RANDOM_NEWTIME_Uni_Mecanum_NEWKCBS_NEWEBKCBS_NEWEBPRRT_NEWPRRT" + str(num_agents))

"""
***************************PROFILING***************************
"""
# import cProfile, pstats, io
# from test_classes import *
# from agent_builders import *

# # collect test classes
# test_classes = [KcbsEbTestClass()]
# # collect agent builders
# agent_builders = [UnicycleBuilder()]
# # instantiate test pipeline 
# tp = TestPipeline(test_classes, agent_builders, test_rounds=50, num_agents=7, master_seed=539)

# with cProfile.Profile() as pr:
#     pr.enable()
#     tp.run()
#     pr.disable()
#     pr.dump_stats('kcbseb_random_10.prof')
#     # s = io.StringIO()
#     # ps = pstats.Stats(pr, stream=s).sort_stats(2)
#     # ps.print_stats()
#     # with open('kcbseb_random_10.txt', 'w') as f:
#     #     print(s.getvalue(), file=f)
# # to put in that web profile thing: flameprof --format=log kcbseb_random_10.prof > kcbseb_random_10.json

# tp.print_stats("Uni_KCBSEB_PROFILED")

# collect test classes
# test_classes = [KcbsTestClass()]
# # instantiate test pipeline 
# tp = TestPipeline(test_classes, agent_builders, test_rounds=50, num_agents=7, master_seed=539)

# with cProfile.Profile() as pr:
#     pr.enable()
#     tp.run()
#     pr.disable()
#     pr.dump_stats('kcbs_random_10.prof')
#     # s = io.StringIO()
#     # ps = pstats.Stats(pr, stream=s).sort_stats(2)
#     # ps.print_stats()
#     # with open('kcbs_random_10.txt', 'w') as f:
#     #     print(s.getvalue(), file=f)
# # to put in that web profile thing: flameprof --format=log kcbs_random_10.prof > kcbs_random_10.json

# tp.print_stats("Uni_KCBS_PROFILED")

# # collect test classes
# test_classes = [PrrtEbTestClass()]
# # instantiate test pipeline 
# tp = TestPipeline(test_classes, agent_builders, test_rounds=50, num_agents=7, master_seed=539)

# with cProfile.Profile() as pr:
#     pr.enable()
#     tp.run()
#     pr.disable()
#     pr.dump_stats('prrteb_random_10.prof')
#     # s = io.StringIO()
#     # ps = pstats.Stats(pr, stream=s).sort_stats(2)
#     # ps.print_stats()
#     # with open('prrteb_random_10.txt', 'w') as f:
#     #     print(s.getvalue(), file=f)
# # to put in that web profile thing: flameprof --format=log prrteb_random_10.prof> prrteb_random_10.json

# tp.print_stats("Uni_PRRTEB_PROFILED")

# # collect test classes
# test_classes = [PrrtTestClass()]
# # instantiate test pipeline 
# tp = TestPipeline(test_classes, agent_builders, test_rounds=50, num_agents=7, master_seed=539)

# with cProfile.Profile() as pr:
#     pr.enable()
#     tp.run()
#     pr.disable()
#     pr.dump_stats('prrt_random_10.prof')
#     # s = io.StringIO()
#     # ps = pstats.Stats(pr, stream=s).sort_stats(2)
#     # ps.print_stats()
#     # with open('prrt_random_10.txt', 'w') as f:
#     #     print(s.getvalue(), file=f)
# # to put in that web profile thing: flameprof --format=log prrt_random_10.prof> prrt_random_10.json

# tp.print_stats("Uni_PRRT_PROFILED")

# collect test classes
# test_classes = [CRRTTestClass()]
# # instantiate test pipeline 
# tp = TestPipeline(test_classes, agent_builders, test_rounds=50, num_agents=7, master_seed=539)

# with cProfile.Profile() as pr:
#     pr.enable()
#     tp.run()
#     pr.disable()
#     pr.dump_stats('crrt_random_10.prof')
#     # s = io.StringIO()
#     # ps = pstats.Stats(pr, stream=s).sort_stats(2)
#     # ps.print_stats()
#     # with open('prrt_random_10.txt', 'w') as f:
#     #     print(s.getvalue(), file=f)
# # to put in that web profile thing: flameprof --format=log crrt_random_10.prof> crrt_random_10.json

# tp.print_stats("Uni_CRRT_PROFILED")

# # collect test classes
# test_classes = [CRRTEBTestClass()]
# # instantiate test pipeline 
# tp = TestPipeline(test_classes, agent_builders, test_rounds=50, num_agents=7, master_seed=539)

# with cProfile.Profile() as pr:
#     pr.enable()
#     tp.run()
#     pr.disable()
#     pr.dump_stats('crrt_eb_random_10.prof')
#     # s = io.StringIO()
#     # ps = pstats.Stats(pr, stream=s).sort_stats(2)
#     # ps.print_stats()
#     # with open('prrt_random_10.txt', 'w') as f:
#     #     print(s.getvalue(), file=f)
# # to put in that web profile thing: flameprof --format=log crrt_eb_random_10.prof> crrteb_random_10.json

# tp.print_stats("Uni_CRRTEB_PROFILED")

"""
***************************PROFILING***************************
"""
