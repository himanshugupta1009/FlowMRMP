import numpy as np
from utils import check_dynamic_collisions_to_end
from sst import SSTNodeMatrix, SSTWitnessMatrix, SST
from numba.typed import List
from numba import types, njit


class EB_SSTNodeMatrix(SSTNodeMatrix):
    def __init__(self, initial_capacity, state_dim, action_dim, max_sub_path_length,
                 max_num_edges_per_node=1000):
        super().__init__(initial_capacity, state_dim, action_dim, max_sub_path_length)
        self.edge_bundle_indices = np.full((initial_capacity, max_num_edges_per_node), 
                                           -1, dtype=np.int64)
        self.edge_bundle_mask = np.full((initial_capacity, max_num_edges_per_node), 
                                        False, dtype=bool)
        self.max_num_edges_per_node = max_num_edges_per_node


    def set_edge_bundle_indices(self, node_index, edge_bundle_indices: np.ndarray):
        l = min(len(edge_bundle_indices), self.max_num_edges_per_node)
        self.edge_bundle_indices[node_index][:l] = edge_bundle_indices[:l]


    def set_edge_bundle_mask(self, node_index, edge_index):
        self.edge_bundle_mask[node_index][edge_index] = True


    def _grow(self):
        super()._grow()
        old_cap = self.edge_bundle_indices.shape[0]
        new_cap = old_cap * 2

        new_edge_bundle_indices = np.full((new_cap, self.max_num_edges_per_node), 
                                         -1, dtype=np.int64)
        new_edge_bundle_mask = np.full((new_cap, self.max_num_edges_per_node), 
                                      False, dtype=bool)
        
        new_edge_bundle_indices[:old_cap] = self.edge_bundle_indices
        new_edge_bundle_mask[:old_cap] = self.edge_bundle_mask
                
        self.edge_bundle_indices = new_edge_bundle_indices
        self.edge_bundle_mask = new_edge_bundle_mask

    def get_edges_for_node(self, node_index):
        return self.edge_bundle_indices[node_index], self.edge_bundle_mask[node_index]


class EB_SST(SST):
    def __init__(self, * , start, goal, goal_radius, env, agent,
                edge_bundle,
                use_fixed_sampling_time=True,
                sampling_time_step=1.5,
                minimum_time_step=0.1,
                max_iter = 10000,
                planning_time = 10.0,
                isvalid_function,
                cost_function,
                reached_goal_function,
                random_point_function,
                translate_function,
                sort_edges_function,
                best_near_radius=2.0,
                prune_radius=0.5,
                max_num_edges_per_node=1000,
                num_skip_edges=50,
                num_random_edges=10,
                eb_kd_tree,
                get_eb_kd_tree_query,
                kd_tree_delta_radius=0.5,
                udf_seed,
                debug_flag=False,
                print_logs=False,
                dynamic_obstacles = List.empty_list(types.Array(types.float64, 2, 'C'))
                ):

        super().__init__(start=start, goal=goal, goal_radius=goal_radius,
                         env=env, agent=agent,
                         use_fixed_sampling_time=use_fixed_sampling_time,
                         sampling_time_step=sampling_time_step,
                         minimum_time_step=minimum_time_step,
                         max_iter=max_iter,
                         planning_time=planning_time,
                         isvalid_function=isvalid_function,
                         cost_function=cost_function,
                         reached_goal_function=reached_goal_function,
                         random_point_function=random_point_function,
                         best_near_radius=best_near_radius,
                         prune_radius=prune_radius,
                         udf_seed=udf_seed,
                         debug_flag=debug_flag,
                         print_logs=print_logs,
                         dynamic_obstacles=dynamic_obstacles
                        )


        self.edge_bundle = edge_bundle
        self.eb_kd_tree = eb_kd_tree
        self.get_eb_kd_tree_query = get_eb_kd_tree_query
        self.kd_tree_delta_radius = kd_tree_delta_radius
        self.num_random_edges = num_random_edges
        self.num_skip_edges = num_skip_edges
        self.max_num_edges_per_node = max_num_edges_per_node
        self.distance_array = np.zeros((self.max_num_edges_per_node,), dtype=np.float64)
        self.random_indices = np.zeros((self.num_random_edges,), dtype=np.int64)
        self.translate = translate_function
        self.sort_edges = sort_edges_function
        
        self._node_matrix = EB_SSTNodeMatrix(initial_capacity=1024,
                                state_dim=self.state_length,
                                action_dim=self.action_length,
                                max_sub_path_length=self.max_sub_path_length,
                                max_num_edges_per_node=max_num_edges_per_node)
        self.type_node_matrix = type(self._node_matrix)
        
    
    def extend_tree(self, parent_node_index, random_point):
        """
        1) Use pre computed edges from edge bundle to extend the tree
        2) Given the new node, first check if valid edges have been found 
        before or not.
        3) If yes, use those edges to extend the tree.
        4) If no, first find those edges using kd-tree 
        and then extend the tree.
        5) Given this new point, sort these edges based on distance to
        the random point.
        6) Pick action and time corresponding to the best edge to 
        extend the tree.
        7) If edge not valid, skip some edges and try the nth best edge.
        8) Keep doing this until a valid edge is found or most of the 
        edges are exhausted.
        9) Update edge_bundle_indices and edge_bundle_mask in the 
        node matrix.   
        """
    
        eb = self.edge_bundle
        p = self.num_skip_edges
        tree_nodes = self._node_matrix
        parent_state = tree_nodes.state[parent_node_index]
        parent_time_elapsed = tree_nodes.time_elapsed[parent_node_index]
        parent_cost = tree_nodes.cost[parent_node_index]

        curr_edge_indices, curr_edge_mask = tree_nodes.get_edges_for_node(parent_node_index)
        if curr_edge_indices[0] == -1:
            #Edges have not been found before for this node.
            #Find edges using search on the kd-tree
            query = self.get_eb_kd_tree_query(parent_state)
            edge_ids = self.eb_kd_tree.radius_query(query, self.kd_tree_delta_radius)
            self._node_matrix.set_edge_bundle_indices(parent_node_index, edge_ids)
            # curr_edge_indices = edge_ids

        """
        1. You have the edge bundle and its indices.
        2. Find the indices of the edges within kd_delta_radius of your query point.
        3. Now, take edges at these indices and sort them based on 
              distance between its final state and the random point.
              Only do this sorting for edges which have not been
                tried before (using edge_bundle_mask).
        4. Return the sorted indices and maybe also how many
            indices to try depending on how many are yet to be tried.
        5. Now, iterate through these sorted indices and try to
                extend the tree.
        """

        # num_edges_from_bundle = len(edge_ids)
        # To-Do: Add an array to count the number of edges possible from 
        # a given node according to the edge bundle.
        sorted_indices, num_valid_edges = self.sort_edges(parent_state,random_point,eb.final_states,
                        curr_edge_indices,curr_edge_mask,self.distance_array)
        
        for idx,x in enumerate(sorted_indices[0:num_valid_edges:p]):
            # print("ID num:" + str(idx) + "Sorted Edge ID : " + str(x))
            action = eb.actions[x]
            timestep = eb.timesteps[x]
            num_record_steps = round(timestep / self.minimum_time_step)
            new_state, path_to_new_state = self.agent.get_next_state(parent_state, action,
                                                    timestep, num_steps=num_record_steps)
            accept_new_node = self.isvalid(path_to_new_state, self.agent.radius, 
                                        self.env.size,
                                        self.static_circular_obstacles,
                                        self.static_rectangular_obstacles,
                                        self.dynamic_agent_obstacles,
                                        self.env.obstacle_buffer,
                                        self.env.boundary_buffer,
                                        parent_time_elapsed,
                                        timestep,
                                        self.minimum_time_step)
            if not accept_new_node:
                if self.debug_flag:
                    print("~~~~~~~~~~Sampled New EB-SST Node is invalid. Trying again!~~~~~~~~~~")
                    print("Invalid Node : ", new_state)       
                continue
            else:
                # print("Node is valid : " + str(new_state))
                # print("Sorted Edge ID : " + str(x))
                reached_goal_flag, goal_distance = self.reached_goal(new_state, self.goal, 
                                                self.goal_radius, self.agent)
                if reached_goal_flag:
                    if check_dynamic_collisions_to_end(new_state, self.agent.radius, 
                                                   self.dynamic_agent_obstacles, 
                                                   self.env.obstacle_buffer,
                                                   parent_time_elapsed + timestep,
                                                   self.minimum_time_step):
                        if self.debug_flag:
                            print("Goal state will collide with high-priority agent. Trying again!")
                        continue

                    edge_cost = self.cost(self.env, self.agent, parent_state, action, 
                                timestep, path_to_new_state)
                    total_elapsed_time = parent_time_elapsed + timestep
                    total_cost = parent_cost + edge_cost
                    new_node_index = self.add_sst_node(new_state, parent_node_index, action, 
                                    timestep, path_to_new_state, total_elapsed_time, total_cost)
                    self.path_found = True
                    if self.debug_flag:
                        print("Goal Reached! Path found for ",self.agent.id)
                    self.goal_node_id = new_node_index
                    self.path_cost = total_cost
                    self.path_time = total_elapsed_time
                    curr_edge_mask[x] = True
                    return new_node_index
                else:
                    if goal_distance < self.threshold:
                        total_elapsed_time = parent_time_elapsed
                        for (index, intermediate_state) in enumerate(path_to_new_state):
                            total_elapsed_time += self.minimum_time_step
                            goal_flag, d = self.reached_goal(intermediate_state, self.goal, 
                                    self.goal_radius, self.agent)
                            if goal_flag:
                                if check_dynamic_collisions_to_end(intermediate_state, self.agent.radius, 
                                                   self.dynamic_agent_obstacles, 
                                                   self.env.obstacle_buffer,
                                                   total_elapsed_time,
                                                   self.minimum_time_step):
                                    if self.debug_flag:
                                        print("Goal state will collide with high-priority agent. Trying again!")
                                    continue

                                modified_edge_time = total_elapsed_time - parent_time_elapsed
                                new_path_to_new_state = path_to_new_state[:index+1]
                                edge_cost = self.cost(self.env, self.agent, parent_state, action,
                                            modified_edge_time, new_path_to_new_state)
                                total_cost = parent_cost + edge_cost
                                new_node_index = self.add_sst_node(intermediate_state, parent_node_index,
                                                        action, modified_edge_time,
                                                        new_path_to_new_state,
                                                        total_elapsed_time, total_cost)
                                self.path_found = True
                                if self.debug_flag:
                                    print("Goal Reached! Path found for ",self.agent.id)
                                self.goal_node_id = new_node_index
                                self.path_cost = total_cost
                                self.path_time = total_elapsed_time
                                curr_edge_mask[x] = True
                                return new_node_index
                            
                    edge_cost = self.cost(self.env, self.agent, parent_state, action,
                                timestep, path_to_new_state)
                    total_elapsed_time = parent_time_elapsed + timestep
                    total_cost = parent_cost + edge_cost
                    new_node_index = self.add_sst_node(new_state, parent_node_index, action, timestep,
                                                path_to_new_state, total_elapsed_time, total_cost)
                    curr_edge_mask[x] = True
                    if self.debug_flag:
                        print("New Node Added to the EB-SST Tree: ", new_node_index)
                    
                    return new_node_index

        #Generate a list of random indices 
        #To-DO: This seems stupid. You can just sample random actions in this case!
        #Or sample a random edge that has not been tried before.
        random_indices = self.rng.integers(0,num_valid_edges,size=self.num_random_edges)

        for idx,x in enumerate(random_indices):
            action = eb.actions[x]
            timestep = eb.timesteps[x]
            num_record_steps = round(timestep / self.minimum_time_step)
            new_state, path_to_new_state = self.agent.get_next_state(parent_state, action,
                                                    timestep, num_steps=num_record_steps)
            accept_new_node = self.isvalid(path_to_new_state, self.agent.radius, 
                                        self.env.size,
                                        self.static_circular_obstacles,
                                        self.static_rectangular_obstacles,
                                        self.dynamic_agent_obstacles,
                                        self.env.obstacle_buffer,
                                        self.env.boundary_buffer,
                                        parent_time_elapsed,
                                        timestep,
                                        self.minimum_time_step)
            if not accept_new_node:
                if self.debug_flag:
                    print("~~~~~~~~~~Sampled New EB-SST Node is invalid. Trying again!~~~~~~~~~~")
                    print("Invalid Node : ", new_state)       
                continue
            else:
                # print("Node is valid : " + str(new_state))
                # print("Random Edge ID : " + str(x))
                reached_goal_flag, goal_distance = self.reached_goal(new_state, self.goal, 
                                                self.goal_radius, self.agent)
                if reached_goal_flag:
                    if check_dynamic_collisions_to_end(new_state, self.agent.radius, 
                                                   self.dynamic_agent_obstacles, 
                                                   self.env.obstacle_buffer,
                                                   parent_time_elapsed + timestep,
                                                   self.minimum_time_step):
                        if self.debug_flag:
                            print("Goal state will collide with high-priority agent. Trying again!")
                        continue
                    edge_cost = self.cost(self.env, self.agent, parent_state, action, 
                                timestep, path_to_new_state)
                    total_elapsed_time = parent_time_elapsed + timestep
                    total_cost = parent_cost + edge_cost
                    new_node_index = self.add_sst_node(new_state, parent_node_index, action, timestep,
                                                path_to_new_state, total_elapsed_time, total_cost)
                    self.path_found = True
                    if self.debug_flag:
                        print("Goal Reached! Path found for ",self.agent.id)
                    self.goal_node_id = new_node_index
                    self.path_cost = total_cost
                    self.path_time = total_elapsed_time
                    curr_edge_mask[x] = True
                    return new_node_index
                else:
                    if goal_distance < self.threshold:
                        total_elapsed_time = parent_time_elapsed
                        for (index, intermediate_state) in enumerate(path_to_new_state):
                            total_elapsed_time += self.minimum_time_step
                            goal_flag, d = self.reached_goal(intermediate_state, self.goal, 
                                    self.goal_radius, self.agent)
                            if goal_flag:
                                if check_dynamic_collisions_to_end(intermediate_state, self.agent.radius, 
                                                   self.dynamic_agent_obstacles, 
                                                   self.env.obstacle_buffer,
                                                   parent_time_elapsed + total_elapsed_time,
                                                   self.minimum_time_step):
                                    if self.debug_flag:
                                        print("Goal state will collide with high-priority agent. Trying again!")
                                    continue
                                
                                modified_edge_time = total_elapsed_time - parent_time_elapsed
                                new_path_to_new_state = path_to_new_state[:index+1]
                                edge_cost = self.cost(self.env, self.agent, parent_state, action,
                                            modified_edge_time, new_path_to_new_state)
                                total_cost = parent_cost + edge_cost
                                new_node_index = self.add_sst_node(intermediate_state, parent_node_index,
                                                        action, modified_edge_time,
                                                        new_path_to_new_state,
                                                        total_elapsed_time, total_cost)
                                self.path_found = True
                                if self.debug_flag:
                                    print("Goal Reached! Path found for ",self.agent.id)
                                self.goal_node_id = new_node_index
                                self.path_cost = total_cost
                                self.path_time = total_elapsed_time
                                curr_edge_mask[x] = True
                                return new_node_index

                            edge_cost = self.cost(self.env, self.agent, parent_state, action,
                                        timestep, path_to_new_state)
                            total_cost = parent_cost + edge_cost
                            total_elapsed_time = parent_time_elapsed + timestep
                            new_node_index = self.add_sst_node(new_state, parent_node_index, action, timestep,
                                                path_to_new_state, total_elapsed_time, total_cost)
                            curr_edge_mask[x] = True

                            if self.debug_flag:
                                print("New Node Added to the EB-SST Tree: ", new_node_index)
                            return new_node_index
                        
        return -1 #Failed to extend the tree using any edge







"""
import sys
sys.path.append('./src')
import numpy as np
from edge_bundle import EdgeBundle

edge_bundle_file_location = 'edge_bundles/eb_unicycle_kinodynamic_TI_edges_100000.npz'
data = np.load(edge_bundle_file_location)
kino_TI_eb_unicycle = EdgeBundle(data, fix_num_edges=30000, use_all_edges=False)

from kd_tree_unicycle import CircularAngleIndexNumba
edge_ids = np.arange(kino_TI_eb_unicycle.num_edges, dtype=np.int64)
thetas = kino_TI_eb_unicycle.start_states[:, 2]  # heading angle θ
kd_tree_kino_TI_eb_unicycle = CircularAngleIndexNumba(thetas, ids=edge_ids)

θq, δ = 6.25, 0.05
ids = kd_tree_kino_TI_eb_unicycle.radius_query(θq, δ)           # np.ndarray of IDs



"""




"""
class A:
    def __init__(self,a):
        self.x = a
    def set_x(self,b):
        self.x[:] = b

a1 = np.array([43,53,64,75,86])
a2 = np.array([21,31,41,51,61])
o = A(a1)
print(o.x)
v = o.x
print(v)
o.set_x(a2)
print(o.x)
print(v)


"""