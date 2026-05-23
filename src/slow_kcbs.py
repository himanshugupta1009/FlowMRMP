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

class KCBSTreeNode:
    def __init__(self, id ,agent_conflicts, paths, trees, path_costs):
        self.id = id
        self.conflicts = agent_conflicts # Dict of conflicts - should have one entry for all agents
        self.agent_paths = paths  # Dictionary of agent_id to path
        self.agent_trees = trees # Dictionary of agent_id to tree
        self.agent_path_costs = path_costs # Total cost of the all the paths
        self.total_cost = sum(path_costs.values())

# function to check collisions when not using a high-level sim
def is_collision_math(first_agent, first_agent_position, second_agent, second_agent_position):
    return (first_agent.check_collision(first_agent_position, second_agent_position) \
                        or second_agent.check_collision(second_agent_position, first_agent_position))

class KCBS:
    def __init__(self, * , env, agents,
                 low_level_planners,
                 max_trials = 1000,
                 planning_time = 100.0,
                 rng_seed = 11,
                 debug_flag=False,
                 print_logs=False,
                 is_collision_function=is_collision_math
                 ):
        self.env = env
        self.num_agents = len(agents)
        self.array_agent_id = np.empty(len(agents),dtype=np.int16) #Numpy array with each agent's id
        self.agents = agents #Dictionary of agents with key as agent's id
        self.low_level_planners = low_level_planners #Dictionary of planners with key as agent's id
        self.max_planning_trials = max_trials
        self.planning_time=planning_time
        self.path_found = False
        self.paths = {}
        self.total_paths_time = 0.0
        self.max_path_time = 0.0
        self.path_cost = float('inf')
        self.path_cbs_node = None
        self.is_collision_function = is_collision_function
        self.cbs_priority_queue = []
        self.tie_breaker = itertools.count()
        self.rng_seed = rng_seed
        self.print_logs = print_logs
        self.debug_flag = debug_flag

        for index, (agent_id, agent) in enumerate(agents.items()):
            self.array_agent_id[index] = agent_id

        for agent_id, planner in self.low_level_planners.items():
            planner.print_logs = self.print_logs
            planner.rng = np.random.default_rng(self.rng_seed)

        self.node_list = [None]*10000


    def sanity_checks(self):
        # Check all agents have different ids
        # Check all agents have the same minimum_time_step
        # Check all agents have non intersecting start and goal regions
        # Check that start and goal regions of all agents don't intersect with obstacles
        return True


    def get_agent_position(self, agent_path_dict, time):
        # max_key = max(agent_path_dict.keys())
        max_key = next(reversed(agent_path_dict))  # Get the last key in the dictionary
        if time >= max_key:
            time = max_key
        return agent_path_dict[time]


    def find_initial_paths(self):
        path_dict,tree_dict,cost_dict = {},{},{}
        for agent_id in self.array_agent_id: #self.agent_ids is a list
            low_level_planner = self.low_level_planners[agent_id]
            low_level_planner.plan_path()
            p = low_level_planner.get_high_resolution_path()
            path_dict[agent_id] = p
            tree_dict[agent_id] = low_level_planner.tree
            cost_dict[agent_id] = low_level_planner.path_cost 

        return path_dict,tree_dict,cost_dict


    def find_all_collisions(self,agent_paths_dict):

        conflict_blocks = []

        for ind_first_agent in range(self.num_agents):
            first_agent_id = self.array_agent_id[ind_first_agent]
            first_agent = self.agents[first_agent_id]
            first_agent_path = agent_paths_dict[first_agent_id]

            for ind_second_agent in range(ind_first_agent+1, self.num_agents):
                second_agent_id = self.array_agent_id[ind_second_agent]
                second_agent = self.agents[second_agent_id]
                second_agent_path = agent_paths_dict[second_agent_id]

                if len(first_agent_path) < len(second_agent_path):
                    longer_path_agent_id = second_agent_id
                else:
                    longer_path_agent_id = first_agent_id

                start_new_block = True
                collision_keys = []
                first_agent_goal_time = next(reversed(first_agent_path))
                second_agent_goal_time = next(reversed(second_agent_path))


                for key in agent_paths_dict[longer_path_agent_id]:
                    first_agent_position = self.get_agent_position(first_agent_path, key)
                    second_agent_position = self.get_agent_position(second_agent_path, key)
 
                    is_collision = self.is_collision_function(first_agent, first_agent_position,
                                                              second_agent, second_agent_position)
                    
                    if(is_collision):
                        if start_new_block:
                            start_key = key
                            collision_keys = []
                            collision_keys.append(key)
                            start_new_block = False
                        else:
                            collision_keys.append(key)
                    else:
                        if not start_new_block:
                            #Create position arrays and add to collision list
                            first_agent_position_array = np.empty(len(collision_keys), 
                                                            dtype=first_agent.state_datatype)
                            second_agent_position_array = np.empty(len(collision_keys),
                                                            dtype=second_agent.state_datatype)
                            for i in range(len(collision_keys)):
                                first_agent_position_array[i] = self.get_agent_position(first_agent_path, 
                                                                                        collision_keys[i])
                                second_agent_position_array[i] = self.get_agent_position(second_agent_path,
                                                                                         collision_keys[i])

                            np_collision_keys = np.array(collision_keys)
                            #Add a conflict for the first agent if the conflict is not at its goal state
                            if(np_collision_keys[0] <= first_agent_goal_time):
                                conflict_block_1 = ConflictBlock(first_agent_id,second_agent_position_array,
                                                                np_collision_keys,second_agent)
                                conflict_blocks.append(conflict_block_1)
                            #Add a conflict for the second agent if the conflict is not at its goal state
                            if(np_collision_keys[0] <= second_agent_goal_time):
                                conflict_block_2 = ConflictBlock(second_agent_id,first_agent_position_array,
                                                                np_collision_keys,first_agent)
                                conflict_blocks.append(conflict_block_2)
                            
                            # conflict_blocks.append(conflict_block_1)
                            # conflict_blocks.append(conflict_block_2)
                            start_new_block = True
                self.env.reset_agent(second_agent)
                # end second agent loop
            self.env.reset_agent(first_agent)
            # end first agent loop         
        return conflict_blocks


    def find_first_collision(self,agent_paths_dict):

        start_conflict = True
        collision_keys = []

        for ind_first_agent in range(self.num_agents):
            first_agent_id = self.array_agent_id[ind_first_agent]
            first_agent = self.agents[first_agent_id]
            first_agent_path = agent_paths_dict[first_agent_id]

            for ind_second_agent in range(ind_first_agent+1, self.num_agents):
                second_agent_id = self.array_agent_id[ind_second_agent]
                second_agent = self.agents[second_agent_id]
                second_agent_path = agent_paths_dict[second_agent_id]

                if len(first_agent_path) < len(second_agent_path):
                    longer_path_agent_id = second_agent_id
                else:
                    longer_path_agent_id = first_agent_id

                first_agent_goal_time = next(reversed(first_agent_path))
                second_agent_goal_time = next(reversed(second_agent_path))


                for key in agent_paths_dict[longer_path_agent_id]:
                    first_agent_position = self.get_agent_position(first_agent_path, key)
                    second_agent_position = self.get_agent_position(second_agent_path, key)

                    is_collision = self.is_collision_function(first_agent, first_agent_position,
                                                              second_agent, second_agent_position)
                    
                    if(is_collision):
                        collision_keys.append(key)
                        start_conflict = False
                    else:
                        if not start_conflict:
                            first_agent_position_array = np.empty((len(collision_keys),first_agent.state_length))
                            second_agent_position_array = np.empty((len(collision_keys),second_agent.state_length))

                            for i in range(len(collision_keys)):
                                first_agent_position_array[i] = self.get_agent_position(first_agent_path, 
                                                                                        collision_keys[i])
                                second_agent_position_array[i] = self.get_agent_position(second_agent_path,
                                                                                            collision_keys[i])

                            np_collision_keys = np.array(collision_keys)
                            collision_start_time = np_collision_keys[0]

                            if self.debug_flag:
                                print(f"Collision detected between agents {first_agent_id} and {second_agent_id} at time {collision_start_time}")

                            conflict_block_1 = ConflictBlock(first_agent_id,second_agent_position_array,
                                                            np_collision_keys,second_agent)
                            conflict_block_2 = ConflictBlock(second_agent_id,first_agent_position_array,
                                                            np_collision_keys,first_agent)
                                
                            return np.array([conflict_block_1,conflict_block_2])

                self.env.reset_agent(second_agent)
                # end second agent loop
            self.env.reset_agent(first_agent)
            # end first agent loop
    
        #No conflicts found          
        return np.empty(0)


    def plan_multi_agent_paths(self):
        
        #Start with no conflicts in the tree!
        no_conflicts_dictionary = {}
        for agent_id in self.array_agent_id:
            # agent_id = agent.id
            no_conflicts_dictionary[agent_id] = []        

        start_time = time.time()
        curr_kcbs_node_counter = 1

        #Find initial paths for all agents
        all_paths_dict, all_trees_dict, all_paths_cost = self.find_initial_paths()
        start_cbs_node = KCBSTreeNode(curr_kcbs_node_counter,no_conflicts_dictionary, all_paths_dict, all_trees_dict,
                                        all_paths_cost)
        curr_kcbs_node_counter += 1
        self.node_list[start_cbs_node.id] = start_cbs_node

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
            parent_conflicts = cbs_node.conflicts

            if self.debug_flag:
                print("Popped node with ID: ", cbs_node.id)
                print("Popped node with cost: ", curr_node_cost)
                print("Popped node with num conflicts per agent: ",
                    [len(conflicts) for agent_id, conflicts in parent_conflicts.items()])
                print("Popped node with paths lengths: ", 
                    [len(path) for path in cbs_node.agent_paths.values()])
                print("Popped node with trees lengths: ", 
                    [len(tree) for tree in cbs_node.agent_trees.values()])
                print("Popped node with paths costs: ", 
                    [cost for cost in cbs_node.agent_path_costs.values()])
                print("Popped node with total cost: ", sum(cbs_node.agent_path_costs.values()))


            if curr_node_cost == float('inf'):
                #Path has not been found yet!

                new_paths_dict = {}
                new_trees_dict = {}
                new_paths_cost_dict = {}

                #Find the agent for which path hasn't been found and try to plan again.
                for agent_id in self.array_agent_id:
                    if( cbs_node.agent_path_costs[agent_id] == float('inf')):
                        agent_path_planner = self.low_level_planners[agent_id]
                        t = cbs_node.agent_trees[agent_id]
                        agent_path_planner.reset_tree(some_existing_tree=t)
                        agent_path_planner.replan_path()
                        new_path = agent_path_planner.get_high_resolution_path()
                        new_paths_dict[agent_id] = new_path
                        new_trees_dict[agent_id] = agent_path_planner.tree
                        new_paths_cost_dict[agent_id] = agent_path_planner.path_cost
                    else:
                        new_paths_dict[agent_id] = cbs_node.agent_paths[agent_id]
                        new_trees_dict[agent_id] = cbs_node.agent_trees[agent_id]
                        new_paths_cost_dict[agent_id] = cbs_node.agent_path_costs[agent_id]

                #Create a new node with the same conflicts as the cbs_node 
                new_node = KCBSTreeNode(cbs_node.conflicts, new_paths_dict, new_trees_dict,
                                       new_paths_cost_dict)
                new_node_cost = new_node.total_cost
                #Add the new node to the priority queue with its corresponding cost
                heapq.heappush(self.cbs_priority_queue, (new_node_cost, next(self.tie_breaker), 
                                                         new_node))

            else:
                #Path has been found for all the agents
                #Check for collisions in the paths of the agents
                # breakpoint()
                curr_cbs_node_conflicts = self.find_first_collision(cbs_node.agent_paths)
                #If no collisions, you have found a solution
                if len(curr_cbs_node_conflicts) == 0:
                    self.path_found = True
                    self.path_cbs_node = cbs_node
                    self.paths = cbs_node.agent_paths 
                    self.path_cost = cbs_node.total_cost
                else:
                    #If collisions are found,
                    #Iterate through all the collisions
                    #For each agent in the collision, add a conflict to that agent
                    #Find the path again for that agent
                    #Add that node to the priority queue with its corresponding cost

                    for conflict in curr_cbs_node_conflicts:
                        #Each conflict is of type ConflictBlock
                        #Create a new node for each conflict
                        #Add the new node to the priority queue
                        #Add the new node to the priority queue with its corresponding cost

                        #Get the agent id of the conflict 
                        conflict_agent_id = conflict.agent_id
                        #Get the current conflict and the previous conflicts for this agent together.
                        old_conflicts = parent_conflicts[conflict_agent_id]
                        new_conflicts = np.append(old_conflicts,conflict)
                        new_conflict_dict = {}
                        for agent_id in self.array_agent_id:
                            # agent = self.agents[agent_id]
                            if agent_id != conflict_agent_id:
                                new_conflict_dict[agent_id] = parent_conflicts[agent_id]
                            else:
                                new_conflict_dict[agent_id] = new_conflicts

                        #Replan the path for this agent
                        conflict_agent_planner = self.low_level_planners[conflict_agent_id]
                        conflict_agent_planner.set_constraints(new_conflicts)
                        # if(cbs_node.id == 2):
                        #     return new_conflicts, conflict_agent_planner 
                        conflict_agent_planner.plan_path_with_constraints(old_conflicts,conflict)
                        new_path_conflict_agent = conflict_agent_planner.get_high_resolution_path()
                        new_tree_conflict_agent = conflict_agent_planner.tree
                        new_path_cost_conflict_agent = conflict_agent_planner.path_cost

                        #Create a new dictionary of paths
                        new_paths_dict = {}
                        new_trees_dict = {}
                        new_paths_cost_dict = {}
                        for agent_id in self.array_agent_id:
                            # agent = self.agents[agent_id]
                            if agent_id == conflict_agent_id:
                                new_paths_dict[agent_id] = new_path_conflict_agent
                                new_trees_dict[agent_id] = new_tree_conflict_agent
                                new_paths_cost_dict[agent_id] = new_path_cost_conflict_agent
                            else:
                                new_paths_dict[agent_id] = cbs_node.agent_paths[agent_id]
                                new_trees_dict[agent_id] = cbs_node.agent_trees[agent_id]
                                new_paths_cost_dict[agent_id] = cbs_node.agent_path_costs[agent_id]

                        #Create a new node with the new conflicts
                        new_node = KCBSTreeNode(curr_kcbs_node_counter, new_conflict_dict, new_paths_dict, new_trees_dict,
                                                new_paths_cost_dict)
                        curr_kcbs_node_counter += 1
                        self.node_list[new_node.id] = new_node
                        new_node_cost = new_node.total_cost
                        
                        #Add the new node to the priority queue
                        heapq.heappush(self.cbs_priority_queue, (new_node_cost, next(self.tie_breaker),
                                                                 new_node))
                        if self.debug_flag:
                            print("Pushed node with ID: ", new_node.id)

            curr_iter += 1
            if (time.time() - start_time >= self.planning_time):
                break

        end_time = time.time()
        if not self.path_found:
            print("MAPF was unsuccessful: Collision free path not found within the given limit of ", 
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