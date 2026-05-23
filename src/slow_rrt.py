import networkx as nx
import numpy as np
import time
from utils import get_dtype_from_input,find_roundoff_decimal_digits,get_nearest_index



class RRTNode:
    def __init__(self, sid, state, parent_id, parent_action, parent_action_duration,
                    path_from_parent, time_so_far, cost):
        self.id = sid
        self.state = state
        self.parent_id = parent_id
        self.parent_action = parent_action
        self.parent_action_duration = parent_action_duration
        self.path_from_parent = path_from_parent
        self.time_elapsed = time_so_far
        self.cost_so_far = cost


"""
while True:
    sample a new point
    find the nearest node in the tree to the sample point
    extend the tree towards the sample point
    if the new node is in collision, continue
    if the new node is not in collision, add the new node to the tree
    if the maximum number of iterations is reached, break
    if the new node is within the goal radius, a path is found, break. 
        
"""

class RRT:
    def __init__(self, * , start, goal, goal_radius, env, agent,
                    use_fixed_sampling_time=True, 
                    sampling_time_step=1.0,
                    minimum_time_step=0.1, 
                    max_iter=1000, 
                    planning_time=10.0,
                    isvalid_function, 
                    cost_function, 
                    reached_goal_function, 
                    random_point_function, 
                    udf_seed = 77,
                    print_flag=False 
                    ):
        
        self.start = start
        self.goal = goal
        self.goal_radius = goal_radius
        self.goal_state_length = len(goal)
        self.env = env
        self.agent = agent
        self.use_fixed_sampling_time = use_fixed_sampling_time #Bool to use fixed sampling time or not
        self.max_sample_T = sampling_time_step #Max time to apply the control for in propogating each edge
        self.minimum_time_step = minimum_time_step #Smallest time duration between consecutive agent positions
        self.max_iter = max_iter #Max number of iterations for RRT
        self.planning_time = planning_time #Time to plan the path in seconds
        self.isvalid = isvalid_function #Function to check if the path is valid
        self.cost = cost_function #Function to calculate the cost of a edge
        self.reached_goal = reached_goal_function
        self.get_random_point = random_point_function
        self.tree = nx.DiGraph()
        self.goal_node_id = None
        self.path_found = False
        self.path_time = 0.0
        self.path_cost = float('inf')
        self.rng_seed = udf_seed
        self.rng = np.random.default_rng(self.rng_seed)
        self.print_flag = print_flag
        self.last_added_node_id = -1

    def get_random_time(self):
        roundoff_digits = find_roundoff_decimal_digits(self.minimum_time_step)
        return round(self.rng.uniform(self.minimum_time_step, self.max_sample_T), roundoff_digits)
    
    def get_fixed_time(self):
        return self.max_sample_T
    
    def get_time(self):
        if self.use_fixed_sampling_time:
            return self.get_fixed_time()
        else:
            return self.get_random_time()

    def num_rrt_nodes(self):
        return len(self.tree.nodes)

    def reset_tree(self, some_existing_tree=None):
        if some_existing_tree == None:
            self.tree = nx.DiGraph()
            self.goal_node_id = None
            self.path_found = False
            self.path_cost = float('inf')
            self.path_time = 0.0
            self.last_added_node_id = -1
        else:
            self.tree = some_existing_tree
            self.goal_node_id = None
            self.path_found = False
            self.path_cost = float('inf')
            self.path_time = 0.0
            last_added_node_id = next(reversed(self.tree._node))
            self.last_added_node_id = last_added_node_id

    def add_rrt_node(self, state, parent_node_id, parent_action, parent_action_duration, 
                        path_from_parent, time_elapsed, cost):
        # new_node_id = len(self.tree.nodes)
        new_node_id = self.last_added_node_id + 1
        new_node = RRTNode(new_node_id, state, parent_node_id, parent_action, parent_action_duration,
                                 path_from_parent, time_elapsed, cost)
        self.tree.add_node(new_node_id, value=new_node)
        self.last_added_node_id = new_node_id
        return new_node_id
     
    def sample_random_point(self):
        """
        TO-DO: Implement a function to sample a random point for every environment and agent 
        that we generate using multiple dispatch from multipledispatch package for this. 
        """
        r = self.rng.uniform(0, 1)
        if r < 0.1:
            random_point = self.goal
        else:
            random_point = self.get_random_point(self.env,self.agent,self.rng)
        return random_point

    def get_nearest_node(self, random_point):
        min_dist = float('inf')
        nearest_rrt_node = None
        nearest_rrt_node_id = None
        for node_id, attr in self.tree.nodes(data=True):
            node_state = attr['value'].state
            # print("Node State: ", node_state)
            dist = self.agent.get_distance(node_state,random_point)
            if dist < min_dist:
                min_dist = dist
                nearest_rrt_node_id = node_id
                nearest_rrt_node = attr['value']
        return nearest_rrt_node_id, nearest_rrt_node
    
    def find_descendants(self, node_id):
        """
        Find all the descendants of the node with id node_id in the tree.
        Since the tree is a directed graph, the descendants are all the nodes that are 
        reachable from node_id.
        Additionally, our RRT tree is only a set of nodes, there are no edges.
        So, we need to iterate and find all the nodes that are reachable from node_id.
        Good thing: Only need to look at the nodes with id > node_id because the nodes are 
        added sequentially with the number of nodes added before it. 

        If there are 100 nodes in the tree, then the node with id 0 is the root node 
        and all the other nodes are its descendants.
        The indices of the nodes go from 0 to 99.
        So, if we want to find the descendants of node_id 19, we need to look at all 
        the nodes with id > 19.
        """

        max_node_id = next(reversed(self.tree._node))
        descendants = {node_id}

        for i in range(node_id+1, max_node_id+1):
            if(i in self.tree.nodes):
                parent_id = self.tree.nodes[i]['value'].parent_id
                if parent_id in descendants:
                    #This means the node is a descendant of the node_id.
                    descendants.add(i)
                #If the node is not a descendant of the node_id, then we can skip it.

        return descendants

    def remove_descendants(self, descendants):
        """
        Remove all the descendants of the node with id node_id from the tree.
        """
        self.tree.remove_nodes_from(descendants)

    def extend_tree(self, parent_node_id, parent_node, random_point):

        """
            * Sample a random action and time duration.
            * Extend the tree towards the random point.
        """

        random_action = self.agent.get_random_action(self.rng)
        random_time = self.get_time()
        num_record_steps = round(random_time/self.minimum_time_step)
        new_state, path_to_new_state = self.agent.get_next_state(parent_node.state, random_action, 
                                                    random_time, num_steps = num_record_steps)
        accept_new_node = self.isvalid(self.env, self.agent, path_to_new_state)
        if not accept_new_node:
            if self.print_flag:
                print("Sampled New RRT Node is invalid. Trying again!")
            return 
        else:
            edge_cost = self.cost(self.env, self.agent, parent_node.state, random_action, 
                                        random_time, path_to_new_state)
            total_elapsed_time = parent_node.time_elapsed + random_time
            total_cost = parent_node.cost_so_far + edge_cost
            new_node_id = self.add_rrt_node(new_state, parent_node_id, random_action, random_time,
                                                path_to_new_state, total_elapsed_time, total_cost)
            reached_goal_flag, goal_distance = self.reached_goal(new_state, self.goal, 
                                                self.goal_radius, self.agent)
            if reached_goal_flag: 
                self.path_found = True
                if self.print_flag:
                    print("Goal Reached! Path found for ",self.agent.id)
                self.goal_node_id = new_node_id
                self.path_cost = total_cost
                self.path_time = total_elapsed_time
            return

    def get_path_to_node_id(self, goal_node_id):

        # Return RRT node ids, agent state and the executed control from start to goal 

        # get state types
        # state_data_type = get_dtype_from_input(self.start)
        state_length = self.agent.state_length
        curr_rng = np.random.default_rng(77)
        control_data_type = get_dtype_from_input(self.agent.get_random_action(curr_rng))

        max_nodes = len(self.tree.nodes)
        path_rrt_node_ids = np.empty(max_nodes, dtype=np.int16)
        # path_states = np.empty(max_nodes, dtype=state_data_type)
        path_states = np.empty((max_nodes, state_length), dtype=np.float64)
        path_controls = np.empty(max_nodes, dtype=control_data_type)
        path_timesteps = np.empty(max_nodes, dtype=np.float64)
        node_id = goal_node_id
        path_length = 0            

        while node_id != -1:
            # print("Node ID: ", node_id)
            rrt_node = self.tree.nodes[node_id]['value']
            path_rrt_node_ids[path_length] = rrt_node.id
            path_states[path_length] = rrt_node.state
            path_controls[path_length] = rrt_node.parent_action
            path_timesteps[path_length] = rrt_node.parent_action_duration
            path_length+=1
            node_id = rrt_node.parent_id

        """
        Note: 
        1) Slicing a numpy array doesn't cause new array allocation. It just returns a view
        of the original array.
        2) Reversing the numpy array using [::-1] doesn't cause new array allocation. 
        It also just returns a view of the original array.
        3) Slicing and reversing the numpy array doesn't cause any allocation!!
        """    

        ids = path_rrt_node_ids[:path_length][::-1]
        states = path_states[:path_length][::-1]
        controls = path_controls[:path_length-1][::-1]
        timesteps = path_timesteps[:path_length-1][::-1]
        return ids, states, controls, timesteps

    def get_path(self):
        # get state types
        state_data_type = get_dtype_from_input(self.start)
        curr_rng = np.random.default_rng(77)
        control_data_type = get_dtype_from_input(self.agent.get_random_action(curr_rng))

        if(self.path_found == False):
            print("Path can't be found because goal hasn't been reached!")
            return np.empty(0,dtype=np.int16), np.empty(0, dtype=state_data_type), \
                    np.empty(0, dtype=control_data_type), np.empty(0, dtype=np.float64)
        else:           
            return self.get_path_to_node_id(self.goal_node_id)

    def get_high_resolution_path(self):
            
        # Return a path with a smaller resolution than the one returned by get_path
        # state_data_type = get_dtype_from_input(self.start)
        # path_dict: Dict[float,state_data_type] = {}
        path_dict = {}
        max_nodes = len(self.tree.nodes)
        path_rrt_node_ids = np.empty(max_nodes, dtype=np.int32)

        if(self.path_found == False):
            print("Path can't be found because goal hasn't been reached!")
            return path_dict
        else:
            node_id = self.goal_node_id
            path_length = 0
            path_time = 0.0            

            while node_id != -1:
                # print("Node ID: ", node_id)
                rrt_node = self.tree.nodes[node_id]['value']
                path_rrt_node_ids[path_length] = rrt_node.id
                path_length+=1
                node_id = rrt_node.parent_id
            
            ids = path_rrt_node_ids[:path_length][::-1]
            start_id = ids[0]
            path_dict[path_time] = self.tree.nodes[start_id]['value'].state
            roundoff_digits = find_roundoff_decimal_digits(self.minimum_time_step)

            for node_id in ids[1:]:
                rrt_node = self.tree.nodes[node_id]['value']
                path_from_parent = rrt_node.path_from_parent
                for point in path_from_parent:
                    path_time = round(path_time + self.minimum_time_step,roundoff_digits)
                    path_dict[path_time] = point

            return path_dict

    def plan_path(self):
        """
        Plan a path using RRT algorithm from scratch.
        """

        self.path_found = False
        self.goal_node_id = None
        self.last_added_node_id = -1
        self.reset_tree()
        first_node_state = self.start 
        self.add_rrt_node(first_node_state, -1, None, None, None, 0.0, 0.0)
    
        curr_num_steps = 0
        start_time = time.time()
        
        while curr_num_steps<=self.max_iter:
            if self.print_flag:
                print("Iteration: ", curr_num_steps)
            
            random_point = self.sample_random_point()
            nearest_node_id, nearest_node = self.get_nearest_node(random_point)
            self.extend_tree(nearest_node_id, nearest_node, random_point)

            if self.path_found or (time.time() - start_time >= self.planning_time):
                break
            curr_num_steps+=1


        end_time = time.time()
        total_time = end_time - start_time
        planning_time_msg = "Total Planning Time"
        if hasattr(self.agent, "id"):
            planning_time_msg += " for agent " + str(self.agent.id)
        print(planning_time_msg + ": ", total_time)

        # return self.get_path(self.goal_node_id)

    def replan_path(self):
        """
        This function is called when planning for a path was attempted before but failed.
        It doesn't regrow the tree from scratch, but expands the tree built during the 
        previous iteration of planning.
        """

        self.path_found = False
        self.goal_node_id = None
        self.path_time = 0.0
        self.path_cost = float('inf')

        print("Replanning the path...")
        curr_num_steps = 0
        start_time = time.time()

        while curr_num_steps<=self.max_iter:
            if self.print_flag:
                print("Iteration: ", curr_num_steps)
            
            random_point = self.sample_random_point()
            nearest_node_id, nearest_node = self.get_nearest_node(random_point)
            self.extend_tree(nearest_node_id, nearest_node, random_point)

            if self.path_found or (time.time() - start_time >= self.planning_time):
                break
            curr_num_steps+=1


        end_time = time.time()
        total_time = end_time - start_time
        planning_time_msg = "Total Planning Time"
        if hasattr(self.agent, "id"):
            planning_time_msg += " for agent " + str(self.agent.id)
        print(planning_time_msg + ": ", total_time)
        