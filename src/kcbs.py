#mapf_lns.py

"""

1) Find paths for all agents from start to goal
2) Check for collisions in all those paths
3) If collision is found, replan a path for the colliding agents from 
delta_t seconds before the collision.
4) Repeat the process until no collisions are found.
5) Return the paths for all agents.

"""

import numpy as np
import heapq
import itertools
from constrainedX import *
import time
from numba.typed import List
from utils import euclidean_distance_satisfaction_numba,euclidean_distance_numba_with_l, \
                    copy_numba_list, DynamicArray
from numba import njit
import copy

class KCBSTreeNode:
    def __init__(self, id, agent_conflicts, paths, trees, path_costs):
        self.id = id
        self.conflicts = agent_conflicts # 1D Numba List of conflicts - one entry for each agent
        self.agent_paths = paths  # 1D Numba List of agent paths - one entry for each agent
        self.agent_trees = trees # 1D Normal Python List of trees - one entry for each agent
        self.agent_path_costs = path_costs # 1D Numba List of Path Costs - one entry for each agent
        self.total_cost = sum(path_costs)


class KCBS:
    def __init__(self, * , env, agents,
                low_level_planners,
                max_trials = 1000,
                planning_time = 100.0,
                minimum_time_step = 0.1,
                clearance_threshold = 0.0,
                rng_seed = 11,
                debug_flag=False,
                print_logs=False,
                prune_tree=False,
                is_collision_function=is_collision_math
                ):
        self.env = env
        self.num_agents = len(agents)
        self.agents = agents #List of agents
        self.low_level_planners = low_level_planners #List of planners
        self.max_planning_trials = max_trials
        self.planning_time=planning_time
        self.path_found = False
        self.paths = {}
        self.total_paths_time = 0.0
        self.max_path_time = 0.0
        self.minimum_time_step = minimum_time_step
        self.path_cost = float('inf')
        self.path_cbs_node = None
        self.is_collision_function = is_collision_function
        self.cbs_priority_queue = []
        self.tie_breaker = itertools.count()
        self.rng_seed = rng_seed
        self.kcbs_rng = np.random.default_rng(self.rng_seed)
        self.print_logs = print_logs
        self.debug_flag = debug_flag
        self.clearance_threshold = clearance_threshold
        self.prune_tree = prune_tree

        self.roundoff_digits = find_roundoff_decimal_digits(self.minimum_time_step)
        seeds_low_level_planners = self.kcbs_rng.integers(0, 10000, size=self.num_agents)

        for i in range(self.num_agents):
            planner = self.low_level_planners[i]
            planner.print_logs = self.print_logs
            # planner.debug_flag = self.debug_flag
            planner.minimum_time_step = self.minimum_time_step
            planner.roundoff_digits = self.roundoff_digits
            planner.rng_seed = seeds_low_level_planners[i]
            planner.rng = np.random.default_rng(seeds_low_level_planners[i])
            planner.dynamic_agent_clearance = self.clearance_threshold
            planner.prune_tree = self.prune_tree

        #Loop through all agents and define an array of agent ids
        self.array_agent_id = np.array([agent.id for agent in agents])

        #Loop through all agents and define an agents_state_lengths
        self.agents_state_length = np.array([agent.state_length for agent in agents], dtype=np.int64)

        #Loop through all agents and define agents_radius
        self.agents_radius = np.array([agent.radius for agent in agents], dtype=np.float64)

        #Define the starting no conflicts list
        self.no_conflicts_list = List()
        dummy_conflict = (np.empty(0), np.empty((0,0)), 0.0)  # Placeholder for conflicts
        for i in range(self.num_agents):
            dummy_conflict_list = List()
            dummy_conflict_list.append(dummy_conflict)
            dummy_conflict_list.pop()
            self.no_conflicts_list.append(dummy_conflict_list)

        #All agents should have the same distance metric state size value
        val = self.agents[0].distance_metric_state_size
        for index in range(1,self.num_agents):
            if self.agents[index].distance_metric_state_size != val:
                raise ValueError("All agents must have the same distance metric state size.")
        self.distance_metric_state_size = val

        self.node_list = DynamicArray(1000,object)

        self.collision_count = np.zeros((self.num_agents,self.num_agents), 
                                        dtype=np.int64)

    def sanity_checks(self):
        # Check all agents have different ids
        # Check all agents have the same minimum_time_step
        # Check all agents have non intersecting start and goal regions
        # Check that start and goal regions of all agents don't intersect with obstacles
        return True

    def plan_multi_agent_paths(self):
        
        start_time = time.time()
        no_conflicts_list = copy_numba_list(self.no_conflicts_list)
        curr_kcbs_node_counter = 1

        #Find initial paths for all agents
        all_paths, all_trees, all_costs = find_initial_paths(self.low_level_planners)
        start_cbs_node = KCBSTreeNode(curr_kcbs_node_counter, no_conflicts_list, all_paths, all_trees, all_costs)
        curr_kcbs_node_counter += 1
        self.node_list.set(start_cbs_node.id, start_cbs_node)

        #Define a priority queue
        start_node_cost = start_cbs_node.total_cost
        heapq.heappush(self.cbs_priority_queue, (start_node_cost,next(self.tie_breaker),
                                                 start_cbs_node))

        #Initialize other necessary variables for the function.        
        curr_iter = 0

        while not self.path_found and curr_iter <= self.max_planning_trials:

            #Pop a node from the priority queue
            if self.print_logs:
                print("*************************************")
                print("Current iteration: ", curr_iter, " at ", time.time() - start_time, " seconds.")

            curr_node_cost, _, cbs_node = heapq.heappop(self.cbs_priority_queue)

            if self.debug_flag:
                print("Popped node with ID: ", cbs_node.id)
                print("Popped node with cost: ", curr_node_cost)
                print("Popped node with num conflicts per agent: ", 
                    [len(conflict) for conflict in cbs_node.conflicts])
                print("Popped node with paths lengths: ", 
                    [len(path) for path in cbs_node.agent_paths])
                print("Popped node with trees lengths: ", 
                    [len(tree[0]) for tree in cbs_node.agent_trees])
                print("Popped node with path costs: ", cbs_node.agent_path_costs)
                print("Popped node with total cost: ", sum(cbs_node.agent_path_costs))

            # return cbs_node
            if curr_node_cost == float('inf'):
                #Path for all the agents have not been found yet!
                #Find the agent for which path hasn't been found and try to plan again.
                for agent_index in range(self.num_agents):
                    if( cbs_node.agent_path_costs[agent_index] == float('inf')):
                        agent_path_planner = self.low_level_planners[agent_index]
                        t = cbs_node.agent_trees[agent_index]
                        agent_path_planner.reset_tree(some_existing_tree=t)
                        agent_path_planner.replan_path()
                        new_path = agent_path_planner.get_high_resolution_path_numpy_array()
                        cbs_node.agent_paths[agent_index] = new_path
                        cbs_node.agent_trees[agent_index] = agent_path_planner.get_tree_structure()
                        cbs_node.agent_path_costs[agent_index] = agent_path_planner.path_cost

                #Modify the existing CBS node to have the same conflicts, but have a new total_cost
                new_total_cost = sum(cbs_node.agent_path_costs)
                cbs_node.total_cost = new_total_cost
                #Add the new node to the priority queue with its corresponding cost
                heapq.heappush(self.cbs_priority_queue, (new_total_cost, next(self.tie_breaker), 
                                                         cbs_node))

            else:
                #Path has been found for all the agents. Check if any of the paths collide.
                curr_cbs_node_conflicts = find_first_collision_numba(cbs_node.agent_paths, 
                            self.agents_state_length,self.agents_radius,self.distance_metric_state_size,
                            self.clearance_threshold,self.roundoff_digits)
                #If no collisions, you have found a solution
                if len(curr_cbs_node_conflicts[0]) == 0:
                    self.path_found = True
                    self.path_cbs_node = cbs_node
                    self.paths = cbs_node.agent_paths 
                    self.path_cost = cbs_node.total_cost
                else:
                    #If collision is found, add a conflict for each agent in the collision
                    #Find the path again for that agent
                    #Add that node to the priority queue with its corresponding cost
                    #Add the new node to the priority queue
                    #Add the new node to the priority queue with its corresponding cost
                    collision_keys, first_agent_index, first_agent_positions, \
                    second_agent_index, second_agent_positions = curr_cbs_node_conflicts
    
                    if(self.debug_flag):
                        print("Collision between agents ", first_agent_index, " and ", second_agent_index, 
                              " at time ", collision_keys)
                        
                    #Increase the correct collision count
                    self.collision_count[first_agent_index, second_agent_index] += 1

                    #Get the radius of the collision agents for constraints
                    first_agent_radius = self.agents_radius[first_agent_index]
                    second_agent_radius = self.agents_radius[second_agent_index]

                    #Copy the current CBS node's conflicts, paths, trees and path costs
                    #to create a new CBS node for the second collision agent.
                    new_cbs_node_conflicts = copy_numba_list(cbs_node.conflicts)
                    new_cbs_node_paths = copy_numba_list(cbs_node.agent_paths)
                    new_cbs_node_trees = copy.copy(cbs_node.agent_trees)
                    new_cbs_node_path_costs = copy_numba_list(cbs_node.agent_path_costs)

                    #For the first agent, to avoid allocation of new lists, we will
                    #use the existing lists in the recently popped cbs_node from the 
                    #priority queue and modify them.
                    curr_first_agent_conflicts = new_cbs_node_conflicts[first_agent_index]
                    new_conflict = (collision_keys, second_agent_positions, second_agent_radius)
                    new_first_agent_conflicts = copy_numba_list(curr_first_agent_conflicts)                    
                    #Add the new conflicts to the first collision agent
                    new_first_agent_conflicts.append( new_conflict )
                    if self.debug_flag:
                        print("New first agent conflicts Number: (id {})".format(first_agent_index), len(new_first_agent_conflicts))

                    #Replan the path for the first collision agent
                    first_agent_planner = self.low_level_planners[first_agent_index]
                    first_agent_planner.set_constraints(new_first_agent_conflicts)
                    curr_first_agent_tree_structure = new_cbs_node_trees[first_agent_index]
                    # if(cbs_node.id == 2):
                    #     return new_first_agent_conflicts, first_agent_planner 
                    first_agent_planner.plan_path_with_constraints(curr_first_agent_tree_structure,
                                                                   new_first_agent_conflicts)
                    new_path_first_agent = first_agent_planner.get_high_resolution_path_numpy_array()
                    #Update the CBS node with new values
                    new_cbs_node_id = curr_kcbs_node_counter
                    curr_kcbs_node_counter += 1
                    new_cbs_node_conflicts[first_agent_index] = new_first_agent_conflicts
                    new_cbs_node_paths[first_agent_index] = new_path_first_agent
                    new_cbs_node_trees[first_agent_index] = first_agent_planner.get_tree_structure()
                    new_cbs_node_path_costs[first_agent_index] = first_agent_planner.path_cost
                    cost_cbs_node_first_conflict = sum(new_cbs_node_path_costs)

                    #Create a new CBS node with the updated conflicts, paths, trees and path costs
                    new_node1 = KCBSTreeNode(new_cbs_node_id,new_cbs_node_conflicts, new_cbs_node_paths, 
                                            new_cbs_node_trees, new_cbs_node_path_costs)
                    self.node_list.set(new_node1.id, new_node1)

                    if(self.debug_flag):
                        print("Replanned path for agent ", first_agent_index, 
                              " with new cost ", first_agent_planner.path_cost)

                    #Copy the current CBS node's conflicts, paths, trees and path costs
                    #to create a new CBS node for the second collision agent.
                    new_cbs_node_conflicts = copy_numba_list(cbs_node.conflicts)
                    new_cbs_node_paths = copy_numba_list(cbs_node.agent_paths)
                    new_cbs_node_trees = copy.copy(cbs_node.agent_trees)
                    new_cbs_node_path_costs = copy_numba_list(cbs_node.agent_path_costs)
                        
                    #Now add conflicts for the second collision agent and replan its path
                    curr_second_agent_conflicts = new_cbs_node_conflicts[second_agent_index]
                    new_conflict = (collision_keys, first_agent_positions, first_agent_radius)
                    new_second_agent_conflicts = copy_numba_list(curr_second_agent_conflicts)                    
                    #Add the new conflicts to the second collision agent
                    new_second_agent_conflicts.append( new_conflict )
                    if self.debug_flag:
                        print("New second agent conflicts Number (id {}): ".format(second_agent_index), len(new_second_agent_conflicts))

                    #Replan the path for the second collision agent
                    second_agent_planner = self.low_level_planners[second_agent_index]
                    second_agent_planner.set_constraints(new_second_agent_conflicts)
                    curr_second_agent_tree_structure = new_cbs_node_trees[second_agent_index]
                    # if(cbs_node.id == 2):
                    #     return new_second_agent_conflicts, second_agent_planner 
                    second_agent_planner.plan_path_with_constraints(curr_second_agent_tree_structure, 
                                                                    new_second_agent_conflicts)
                    new_path_second_agent = second_agent_planner.get_high_resolution_path_numpy_array()
                    #Update the new CBS node with new values
                    new_cbs_node_id = curr_kcbs_node_counter
                    curr_kcbs_node_counter += 1
                    new_cbs_node_conflicts[second_agent_index] = new_second_agent_conflicts
                    new_cbs_node_paths[second_agent_index] = new_path_second_agent
                    new_cbs_node_trees[second_agent_index] = second_agent_planner.get_tree_structure()
                    new_cbs_node_path_costs[second_agent_index] = second_agent_planner.path_cost
                    cost_cbs_node_second_conflict = sum(new_cbs_node_path_costs)

                    if(self.debug_flag):
                        print("Replanned path for agent ", second_agent_index, 
                              " with new cost ", second_agent_planner.path_cost)

                    #Create a new CBS node with the updated conflicts, paths, trees and path costs
                    new_node2 = KCBSTreeNode(new_cbs_node_id, new_cbs_node_conflicts, new_cbs_node_paths, 
                                            new_cbs_node_trees, new_cbs_node_path_costs)
                    self.node_list.set(new_node2.id, new_node2)

                    if(self.debug_flag):
                        print("Adding two new nodes to the priority queue with costs ",
                              cost_cbs_node_first_conflict, " and ", cost_cbs_node_second_conflict)
                        
                    #Add the two new nodes to the priority queue
                    heapq.heappush(self.cbs_priority_queue, (cost_cbs_node_first_conflict, next(self.tie_breaker),
                                                                new_node1))
                    heapq.heappush(self.cbs_priority_queue, (cost_cbs_node_second_conflict, next(self.tie_breaker),
                                                                new_node2))
                    if self.debug_flag:
                        print("Pushed node with ID: ", new_node1.id)
                        print("Pushed node with ID: ", new_node2.id)

            curr_iter += 1
            if (time.time() - start_time >= self.planning_time):
                break

        end_time = time.time()
        if not self.path_found:
            print("MRMP with KCBS was unsuccessful: Collision free path not found within the given limit of ", 
                  self.planning_time, "seconds!")
        
        return self.path_found, self.paths, self.path_cost, end_time-start_time


"""

1) First find path of all agents from start to goal
2) Start a while loop until some criterion is met
    Pop a node from the priority queue
    2.1) Find all collisions in the paths of the agents
    2.2) If no collisions, you have found a solution : return the paths
    2.3) If collisions are found,
        2.3.1) Iterate through all the collisions
        2.3.2) For each agent in the collision, add a conflict to that agent
        2.3.3) Find the path again for that agent
        2.3.4) Add that node to the priority queue with its corresponding cost


"""

@njit 
def get_position(agent_path, index):
    if index >= len(agent_path):
        return agent_path[-1]
    return agent_path[index]

@njit
def find_first_collision_numba(agent_paths,agent_state_lengths,
                               agent_radiuses,distance_metric_state_size,
                               dynamic_agent_clearance=0.0,
                               roundoff_digits=1):

    start_conflict = True
    starting_collision_key = 0
    num_agents = len(agent_paths)

    for ind_first_agent in range(num_agents):
        first_agent_path = agent_paths[ind_first_agent]
        first_agent_state_length = agent_state_lengths[ind_first_agent]
        first_agent_radius = agent_radiuses[ind_first_agent]

        for ind_second_agent in range(ind_first_agent+1, num_agents):
            second_agent_path = agent_paths[ind_second_agent]
            second_agent_state_length = agent_state_lengths[ind_second_agent]
            second_agent_radius = agent_radiuses[ind_second_agent]

            # print("Checking collision between agents ", ind_first_agent, " and ", ind_second_agent)
            # if not (ind_first_agent == 1 and ind_second_agent == 5):
            #     continue

            if len(first_agent_path) < len(second_agent_path):
                longer_path_len = len(second_agent_path)
            else:
                longer_path_len = len(first_agent_path)
            threshold = first_agent_radius + second_agent_radius + dynamic_agent_clearance
            # print("threshold: ", threshold)
            # print("longer_path_len: ", longer_path_len)
            for index in range(longer_path_len):
                first_agent_position = get_position(first_agent_path, index)
                # print("First agent position at index ", index, ": ", first_agent_position)
                second_agent_position = get_position(second_agent_path, index)
                # print("Second agent position at index ", index, ": ", second_agent_position)
                
                dist = euclidean_distance_numba_with_l(first_agent_position, second_agent_position,
                                                       distance_metric_state_size)
                is_collision = dist <= threshold
                # print("is_collision: ", is_collision, " dist: ", dist, " index: ", index)
                if(is_collision and index != longer_path_len - 1):
                    if (start_conflict):
                        starting_collision_key = index
                        start_conflict = False
                elif(is_collision and index == longer_path_len - 1) or \
                            (not is_collision and not start_conflict):
                        #Found the first index after collison has ended or reached the end of the
                        #path with a collision.
                        #Create position arrays and add to collision list
                        # print("Collision started at index ", starting_collision_key)
                        # print("Collision ended at index ", index)
                        # print("Starting collision key: ", starting_collision_key)
                        if is_collision and start_conflict:
                            #This condition is reached when collision occurs only at the
                            #last index of the longer_path.
                            starting_collision_key = index
                            start_conflict = False
                            collision_length = 1
                        else:
                            collision_length = index - starting_collision_key
                        first_agent_position_array = np.empty((collision_length,first_agent_state_length))
                        # first_agent_position_array[:] = first_agent_path[starting_collision_key:index]
                        for i in range(collision_length):
                            first_agent_position_array[i] = get_position(first_agent_path, starting_collision_key + i)
                        second_agent_position_array = np.empty((collision_length,second_agent_state_length))
                        # second_agent_position_array[:] = second_agent_path[starting_collision_key:index]
                        for i in range(collision_length):
                            second_agent_position_array[i] = get_position(second_agent_path, starting_collision_key + i)
                        #Create collision keys
                        collision_keys = np.empty(collision_length, dtype=np.float64)
                        for i in range(collision_length):
                            collision_keys[i] = round(0.1*(starting_collision_key + i), roundoff_digits)

                        return collision_keys, ind_first_agent, first_agent_position_array, \
                                ind_second_agent, second_agent_position_array

    #No conflicts found
    return np.empty(0), np.int64(-1), np.empty((0,0),dtype=np.float64), np.int64(-1), \
                    np.empty((0,0),dtype=np.float64)


def find_initial_paths(low_level_planners):
    paths = List()
    trees = []
    costs = List()

    for i in range(len(low_level_planners)):
        low_level_planner = low_level_planners[i]
        low_level_planner.plan_path()
        p = low_level_planner.get_high_resolution_path_numpy_array()
        paths.append(p)
        trees.append(low_level_planner.get_tree_structure())
        costs.append(low_level_planner.path_cost)

    return paths, trees, costs


def check_high_resolution_paths_collision_free(high_resolution_paths,agents,
                                distance_metric_state_size=None,
                                dynamic_agent_clearance=0.0,
                                roundoff_digits=1,):

    if distance_metric_state_size is None:
        distance_metric_state_size = agents[0].distance_metric_state_size
    for agent in agents:
        if agent.distance_metric_state_size != distance_metric_state_size:
            raise ValueError(
                "All agents must have the same distance_metric_state_size "
                "for collision checking.")
    
    agent_paths = List()
    for path in high_resolution_paths:
        agent_paths.append(np.asarray(path, dtype=np.float64))

    agent_state_lengths = np.array(
        [agent.state_length for agent in agents],
        dtype=np.int64,
    )

    agent_radiuses = np.array(
        [agent.radius for agent in agents],
        dtype=np.float64,
    )

    collision_keys, first_agent, first_positions, second_agent, second_positions = (
        find_first_collision_numba(
            agent_paths,
            agent_state_lengths,
            agent_radiuses,
            distance_metric_state_size,
            dynamic_agent_clearance=dynamic_agent_clearance,
            roundoff_digits=roundoff_digits,
        )
    )

    return {
        "collision_free": first_agent == -1,
        "collision_times": collision_keys,
        "first_agent": int(first_agent),
        "second_agent": int(second_agent),
        "first_positions": first_positions,
        "second_positions": second_positions,
    }

"""
# Trying out some multithreading and multiprocessing for find_initial_paths

from concurrent.futures import ProcessPoolExecutor
from numba.typed import List

def run_planner(planner):
    planner.plan_path()
    path = planner.get_high_resolution_path_numpy_array()
    tree = planner.tree  # Be cautious: must be picklable!
    cost = planner.path_cost
    return path, tree, cost

def find_initial_paths_parallel(low_level_planners):
    paths = []
    trees = []
    costs = []

    with ProcessPoolExecutor() as executor:
        results = list(executor.map(run_planner, low_level_planners))

    for path, tree, cost in results:
        paths.append(path)
        trees.append(tree)
        costs.append(cost)

    return paths, trees, costs


planners = []

for i in range(num_agents):
    planners.append( planner_function(starts[i],goals[i],0.5,agents[i],env) )

p,t,c = find_initial_paths(planners)

find_initial_paths_parallel(planners)
"""
