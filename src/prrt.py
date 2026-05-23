from rrt import RRT
from Environments import AgentObstacle
import time
from utils import check_dynamic_collisions_to_end, check_dynamic_collisions_to_end_3d
from numba.typed import List
from numba import types
import numpy as np

class PRRT(RRT):

    def __init__(self, *args, **kwargs):
        """
        Priority RRT (pRRT) planner class. Inherits from RRT class.

        Each agent plans its path individually in a priority order. Subsequent agents 
        must plan using the previous agents' positions along their paths as obstacles. 

        See RRT class for parameters.
        """
        super().__init__(*args, **kwargs)

        # set the appropriate dynamic obstacle collision checkers
        if self.distance_metric_state_size == 2:
            self.dynamic_col_checker_to_end = check_dynamic_collisions_to_end
        elif self.distance_metric_state_size == 3:
            self.dynamic_col_checker_to_end = check_dynamic_collisions_to_end_3d
        else:
            raise NotImplementedError("PRRT dynamic collision checking not implemented for position state size ",
                                      self.distance_metric_state_size)        

    def extend_tree(self, parent_node_id, parent_node, random_point):
        """
        Sample a random action and time duration.
        Extend the tree towards the random point.

        :MAINT: This is extended from the base RRT class as 
            obstacles for pRRT are time-dependent 

        parent_node_id: Tree node id for the parent node, or the
            state to propagate from 
        parent_node: the RRT node representation of the parent node
        random_point: not used for non-edgebundle planning
        """
        best_candidate = self._select_best_extension_candidate(
                        parent_node,random_point)

        if best_candidate is None:
            if self.debug_flag:
                print("~~~~~~~~~~All extension trials invalid~~~~~~~~~~")
                print("~~~~~~~~~~Sampled New RRT Node is invalid. Trying again!~~~~~~~~~~")
            return 
        else:
            new_state, path_to_new_state, random_action, random_time = best_candidate

            reached_goal_flag, goal_distance = self.reached_goal(new_state, self.goal, 
                                                            self.goal_radius, self.agent)
            if reached_goal_flag:

                if self.dynamic_col_checker_to_end(new_state, self.agent.radius, 
                                                   self.dynamic_agent_obstacles, 
                                                   self.dynamic_agent_clearance,
                                                   parent_node.time_elapsed + random_time,
                                                   self.minimum_time_step):
                    if self.debug_flag:
                        print("Goal state will collide with high-priority agent. Trying again!")
                    return

                edge_cost = self.cost(self.env, self.agent, parent_node.state, random_action, 
                            random_time, path_to_new_state)
                total_elapsed_time = parent_node.time_elapsed + random_time
                total_cost = parent_node.cost_so_far + edge_cost

                new_node_id = self.add_rrt_node(new_state, parent_node_id, random_action, random_time,
                                                path_to_new_state, total_elapsed_time, total_cost)
                self.path_found = True
                if self.debug_flag:
                    print("Goal Reached! Path found for ",self.agent.id)
                self.goal_node_id = new_node_id
                self.path_cost = total_cost
                self.path_time = total_elapsed_time
                return
            else:
                if goal_distance < self.threshold:
                    total_elapsed_time = parent_node.time_elapsed
                    for (index, intermediate_state) in enumerate(path_to_new_state):
                        total_elapsed_time += self.minimum_time_step
                        goal_flag, d = self.reached_goal(intermediate_state, self.goal, 
                                                         self.goal_radius, self.agent)
                        if goal_flag:
                            if self.dynamic_col_checker_to_end(intermediate_state, self.agent.radius, 
                                                   self.dynamic_agent_obstacles, 
                                                   self.dynamic_agent_clearance,
                                                   total_elapsed_time, 
                                                   self.minimum_time_step):
                                if self.debug_flag:
                                    print("Goal state will collide with high-priority agent. Trying again!")
                                return
                            
                            modified_edge_time = total_elapsed_time - parent_node.time_elapsed
                            new_path_to_new_state = path_to_new_state[:index+1]
                            edge_cost = self.cost(self.env, self.agent, parent_node.state, random_action,
                                                modified_edge_time, new_path_to_new_state)
                            total_cost = parent_node.cost_so_far + edge_cost

                            new_node_id = self.add_rrt_node(intermediate_state, parent_node_id, random_action, 
                                                            modified_edge_time, new_path_to_new_state, 
                                                            total_elapsed_time, total_cost)
                            self.path_found = True
                            if self.debug_flag:
                                print("Goal Reached! Path found for ",self.agent.id)
                            self.goal_node_id = new_node_id
                            self.path_cost = total_cost
                            self.path_time = total_elapsed_time
                            return
                
            edge_cost = self.cost(self.env, self.agent, parent_node.state, random_action, 
                                    random_time, path_to_new_state)
            total_elapsed_time = parent_node.time_elapsed + random_time
            total_cost = parent_node.cost_so_far + edge_cost

            new_node_id = self.add_rrt_node(new_state, parent_node_id, random_action, random_time,
                                            path_to_new_state, total_elapsed_time, total_cost)
            if self.debug_flag:
                print("New Node Added to the RRT Tree: ", new_node_id)


            return
                
    @staticmethod
    def plan_multi(*, agents, starts, goals, goal_radii, env,
                    use_fixed_sampling_time=True, sampling_time_step=1.0, max_iter=float('inf'), 
                    planning_time=10.0,isvalid_function, cost_function, reached_goal_function, 
                    random_point_function, udf_seed = 77, obs_type = AgentObstacle, print_logs=False,
                    debug_flag=False, num_extension_trials=1):
        """Perform Priority RRT (pRRT)
        Each agent plans its path individually in a priority order. Subsequent agents 
        must plan using the previous agents' positions along their paths as obstacles. 

        Args:
            agents (list(agent objects)): list of agents 
            starts (list(agent_state_type)): agent starts
            goals (list(tuple(x, y))): list of agent goal centers
            goal_radii (list(float)): list of agent goal radii
            env: environment object
            isvalid_function (bool): List of functions for each agent that return true if a path is valid
                for an agent 
                (agent_state_type path_to_new_state, float agent_radius, tuple env_size,
                    list circ_obs, list rect_obs, list dyn_obs,
                    np.array limit_indices, np.array limit_values,
                    float obstacle_buffer, float boundary_buffer,
                    float start_time, float time_duration,
                    float dt_per_step=0.1) -> bool  
            cost_function (float): List of functions for each agent that return the cost an agent 
                incurs over a path
                (env object, agent object, agent_state_type start state, agent_action_type random_action,
                    float time_delta, list(agent_state_type) path_to_new_state) -> float
            reached_goal_function (bool): List of functions for each agent that returns true if the agent 
                has reached its goal, false else. 
                (agent_state_type state, tuple(x float, y float) goal center, float goal radius, 
                    agent object) -> bool  
            random_point_function (tuple(x float, y float)): List of functions for each agent that 
                generate a new point in the environment 
                (env object, list circular obstacles, list rectangular obstacles, rng object) -> tuple(x float, y float)
            use_fixed_sampling_time (bool, optional): Don't use random timesteps. Defaults to True.
            sampling_time_step (float, optional): Max sampling time to use. Defaults to 1.0.
            max_iter (int, optional): Maximum number of planning iterations before failure. Defaults to 1000.
            planning_time (float, optional): maximum time to plan before failure. Defaults to 10.0.
            udf_seed (int, optional): local rng seed. Defaults to 77.
            obs_type (Obstacle, optional): Type of obstacle to use to represent agents that have found
                goals for other agents to avoid. Defaults to AgentObstacle.
            print_logs (bool, optional): Print information from RRT process. Defaults to False.
            debug_flag (bool, optional): Print even more information. Defaults to false

        Returns:
            tuple(list, list, list, list, list, float): lists of:
                Successful path node ids for each agent
                Successful path state tuples for each agent
                Successful path control inputs for each agent
                Successful path timesteps, i.e. the time delta between each node in the path, for each agent
                    :MAINT: The timesteps are NOT cumulative!
                The pRRT object for each agent.
                As well as the total planning time in fractional seconds.
        """

        # list of return values for each agent
        paths = []
        states = []
        controls = []
        timesteps = []
        rrts = []

        dyn_obs = List.empty_list(types.Array(types.float64, 2, 'C'))

        seed_rng = np.random.default_rng(udf_seed)
        planner_seeds = seed_rng.integers(0, np.iinfo(np.int32).max, size=len(agents))

        start_time = time.time()
        # iterate over each agent, planning for it
        for i, agent, start, goal, goal_radius in zip(range(len(agents)), agents, starts, goals, goal_radii):
            if(print_logs):
                print("New Agent at", (time.time() - start_time), "seconds.")
            env.add_agent(agent, goal=(goal, goal_radius))

            # create a new PRRT object 
            rrt = PRRT( start=start, goal=goal, goal_radius=goal_radius, 
                env = env, agent=agent, 
                use_fixed_sampling_time=use_fixed_sampling_time,
                sampling_time_step=sampling_time_step,
                max_iter = max_iter, planning_time=planning_time - (time.time() - start_time),         
                isvalid_function=isvalid_function[i], 
                cost_function=cost_function[i],
                random_point_function=random_point_function[i], 
                reached_goal_function = reached_goal_function[i],
                udf_seed = int(planner_seeds[i]),
                dynamic_obstacles=dyn_obs,
                print_logs=print_logs,
                debug_flag=debug_flag,
                num_extension_trials=num_extension_trials
                )
            # Find the path with the PRRT object
            rrt.plan_path()

            # Get path information for each agent 
            path_state_ids, path_states, path_controls, path_timesteps = rrt.get_path()
            #Q from HG: Why do we calculate this here? Its additional complexity and is not really needed AFAIK!
            paths.append(path_state_ids)
            states.append(path_states)
            controls.append(path_controls)
            timesteps.append(path_timesteps)
            rrts.append(rrt)

            # add new agent obstacle if there was a path found for this 
            # agent 
            if(rrt.path_found and time.time() - start_time <= planning_time):
                dyn_obs.append(obs_type(agent, rrt).to_np())
            else:
                end_time = time.time() - start_time 
                print("PRRT failed to find a path for agent", i)
                print("Failed to solve the current MRMP problem using pRRT.")
                return ([], [], [], [], [], end_time)

        end_time = time.time() - start_time 
        return (paths, states, controls, timesteps, rrts, end_time)
