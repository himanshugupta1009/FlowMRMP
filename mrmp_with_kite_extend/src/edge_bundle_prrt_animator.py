import matplotlib.pyplot as plt
import networkx as nx
from prrt_eb import EdgeBundlePRRT
from printer import RRTPrinter

class EdgeBundleRRTAnimator(EdgeBundlePRRT):
    def __init__(self, * , start, goal, goal_radius, env, agent, edge_bundle,
                    use_fixed_sampling_time=True, 
                    sampling_time_step=1.0,
                    minimum_time_step=0.1,
                    max_iter=1000,
                    num_random_edges=10, 
                    num_skip_edges=10, 
                    planning_time=np.Inf, 
                    isvalid_function,
                    cost_function,
                    reached_goal_function, 
                    translate_function,
                    random_point_function,
                    debug_flag=False,
                    print_logs=False,
                    udf_seed = 77, 
                    print_eb = True):
        """
        Initiate a new Edge Bundle RRT planner with interactive rendering. 

        Args:
            start (agent_state_type): Agent start position
            goal (tuple(x, y)): Agent goal region center 
            goal_radius (float): Agent goal region radius
            env: Environment object
            agent: Agent object
            edge_bundle (EdgeBundle): edge bundle for agent
            isvalid_function (bool): A function that returns true if a state is valid
                for an agent 
                (environment, agent, list(agent_state_type) path_to_new_state, step_time, path_timestep) 
                    -> bool is_valid_state, 
            cost_function (float): returns the cost an agent incurs over a path
                (env object, agent object, agent_state_type start state, agent_action_type random_action,
                    float time_delta, list(agent_state_type) path_to_new_state) -> float
            reached_goal_function (bool): a function that returns true if the agent has reached its goal,
                false else. 
                (agent_state_type state, tuple(x float, y float) goal center, float goal radius, 
                    agent object) -> bool  
            random_point_function (tuple(x float, y float)): generates a new point in the environment 
                (env object, agent object, rng object) -> tuple(x float, y float)
            translate_function (agent_state_type): List of functions for each agent that returns
                the end point from an edge bundle translated to start from a current agent state
                (agent_state_tuple, agent_state_tuple) -> agent_state_tuple
            use_fixed_sampling_time (bool, optional): _description_. Defaults to True.
            use_fixed_sampling_time (bool, optional): Don't use random timesteps. Defaults to True.
            minimum_time_step (float, optional): Minimum path timestep. Defaults to 0.1.
            max_iter (int, optional): Maximum number of planning iterations before failure. Defaults to 1000.
            num_random_edges (int, optional): Number of random edges to select when extending 
                Tree. Defaults to 10.
            num_skip_edges (int, optional): Number of edge bundles to skip while iterating through
                sorted edge bundles. Defaults to 10.
            planning_time (float, optional): Max planning time second. Defaults to np.Inf.
            debug_flag (bool, optional): Print various debugging info to the console. Defaults to True.
            print_logs (bool, optional): Print logs to the console when path is found. Defaults to True.
            udf_seed (int, optional): local rng seed. Defaults to 77
            print_eb (bool, optional): Render edge bundles during iteration. Defaults to True.
        """
        super().__init__(start=start, goal=goal, goal_radius=goal_radius, env=env, agent=agent, 
                    edge_bundle=edge_bundle,
                    use_fixed_sampling_time=use_fixed_sampling_time, 
                    sampling_time_step=sampling_time_step,
                    minimum_time_step=minimum_time_step,
                    max_iter=max_iter,
                    num_random_edges=num_random_edges, 
                    num_skip_edges=num_skip_edges, 
                    planning_time=planning_time, 
                    isvalid_function=isvalid_function,
                    cost_function=cost_function,
                    reached_goal_function=reached_goal_function, 
                    translate_function=translate_function,
                    random_point_function=random_point_function,
                    debug_flag=debug_flag,
                    print_logs=print_logs,
                    udf_seed=udf_seed)
        # rendering frame
        self.fig, self.ax = plt.subplots()
        # holds the last list of sorted edges
        self.sorted_edges = [] 
        # Render edge bundles during iteration
        self.print_eb = print_eb
        # holds the last parent node id propagated from 
        self.last_parent = None
        # holds the last random point generated
        self.last_random_point = None


    def plot_tree(self):
        """
        Display the environment with the current tree in a window, including edge
        bundles   
        """
        # set up window
        self.ax.clear()
        self.ax.set_xlim(0, self.env.size[0])
        self.ax.set_ylim(0, self.env.size[1])
        self.ax.set_aspect('equal', adjustable='box')
        
        # gather nodes as list
        nodes = [] 
        for node in self.tree.nodes(data=True):
            nodes.append(node[1].get('value'))
        
        # render start
        self.ax.scatter(*self.start[0:2], color='green', s=100, label='Start')
        
        RRTPrinter.print_goal(self.ax, self.goal, self.goal_radius)

        # render obstacles
        RRTPrinter.print_obs(self.ax, self.env.obstacles, self.env.obstacle_buffer)

        if self.print_eb:
            # render edge bundles considered if requested
            p = self.num_skip_edges
            eb = self.edge_bundle
            for idx, x in enumerate(self.sorted_edges[::p]):
                parent_node = self.tree.nodes(data=True)[self.last_parent].get('value')
                action = eb.actions[x]
                timestep = eb.timesteps[x]
                num_record_steps = round(timestep/self.minimum_time_step)
                _, path = self.agent.get_next_state(parent_node.state, action, 
                                                            timestep, num_steps = num_record_steps)
                parent_state = parent_node.state
                xs = [parent_state[0]] + [i[0] for i in path]
                ys = [parent_state[1]] + [i[1] for i in path]
                self.ax.plot(xs, ys, linestyle='-', color='xkcd:mauve') #, markersize=.1)

        if(len(nodes) > 1):
            # if this isn't the first node, render the path to it
            RRTPrinter.print_tree_ani(self.ax, nodes[:-1], all_nodes=nodes)
        RRTPrinter.print_path_ani(self.ax, [nodes[-1]], col='blue')
        # render agent
        RRTPrinter.print_agent(self.ax, self.agent, nodes[-1].state[:2])


        if self.last_random_point is not None:
            # show the goal point if it exists
            self.ax.scatter(*self.last_random_point[0:2], color='xkcd:dusty pink', s=100, label='Random Point')
        
        plt.legend()
        plt.draw()
        while True:
            key = plt.waitforbuttonpress()
            if key is not None:  # wait for button press/mouse click
                break
        plt.pause(0.01)
    
    def add_rrt_node(self, state, parent_node_id, action, timestep, path_to_new_state, total_elapsed_time, total_cost):
        """
        Adds a new RRT node to the tree

        :MAINT: Overloaded to catch last parent

        Args:
            states (state_agent_type): New state
            parent_node (int): Parent node id
            action (agent_control_type): Action from the parent state 
                to the new state
            parent_action_duration (float): time taken from parent state to new state
            path_from_parent (list(agent_state_type)): Path the agent
                takes from the parent state to the new state
            cost (float): Cost for agent's new path path 
            time (float): current time at new state 

        Returns:
            int: The new node ID
        """
        node_id = super().add_rrt_node(state, parent_node_id, action, timestep, path_to_new_state, total_elapsed_time, total_cost)
        # capture the last parent node id
        self.last_parent = parent_node_id
        self.plot_tree()
        return node_id
    
    def sort_edges(self, closest_tree_point, random_point):
        """
        Sort edges according to distance to random point

        :MAINT: Overloaded to catch list of sorted eb, print info to console

        Args:
            closest_tree_point (agent_state_type): point propagating from
            random_point (agent_state_type): point propagating torwards

        Returns:
            list(int): indices into edge bundles list, sorted by distance to random_point
                after propagating from closest_tree_point increasing 
        """
        eb = self.edge_bundle
        # capture last random point
        self.last_random_point = random_point
        for i in range(eb.num_edges):
            potential_new_point = self.translate(closest_tree_point,eb.final_states[i])
            dist = self.agent.get_distance(potential_new_point,random_point)
            self.distance_array[i] = dist

        sorted_indices = np.argsort(self.distance_array)
        potential_new_point = self.translate(closest_tree_point,eb.final_states[sorted_indices[0]])
        print(f"Closest tree point: {closest_tree_point}")
        print(f"Closest edge: {sorted_indices[0]} with distance: {self.distance_array[sorted_indices[0]]:.2f}")
        print(f"Closest edge point: {potential_new_point}")
        print(f"Best Edge from the Bundle: {eb.final_states[sorted_indices[0]]}")
        # capture sorted indices 
        self.sorted_edges = sorted_indices
        return sorted_indices


# Example usage

from Environments import *
from Agents import UniCycle, Mecanum
from mapf_env_square_agent_unicycle import point_translate_function, is_new_node_valid, get_cost, get_random_point, agent_reached_goal
import matplotlib.pyplot as plt


from edge_bundle import EdgeBundle
import numpy as np
# edge_bundle_file_location = 'edge_bundles/eb_unicycle_edges_100000.npz' 
edge_bundle_file_location = 'edge_bundles/eb_mecanum_r1_s2_edges-100000_20250507-004006.npz' 
data = np.load(edge_bundle_file_location)
edge_bundle = EdgeBundle(data, fix_num_edges=1000)

start = (5.0, 2.0, 0.0)
goal = (25.0, 25.0, 0.0)
goal_radius = 2.0

obstacles = [
            CircularObstacle2D(10, 10, 2),
            CircularObstacle2D(16, 25, 3),
            CircularObstacle2D(20, 5, 2),
            CircularObstacle2D(35, 15, 2),
            CircularObstacle2D(30, 34, 4),
            CircularObstacle2D(25, 15, 4),
            CircularObstacle2D(7, 19, 4),
            CircularObstacle2D(16, 16, 2),
            CircularObstacle2D(33, 4, 2),
            CircularObstacle2D(8, 34, 3),
            CircularObstacle2D(20, 32, 2),
            CircularObstacle2D(31, 24, 3),
            ]   
                    
env = SquareEnvironment(40, 40, obstacles)
# agent = UniCycle(agent_id = 1, 
#                  max_speed = 2.0,
#                  max_omega= math.pi/2,
#                  radius = 1.0,
#                  rng_seed= 77)
agent = Mecanum(agent_id=1, max_speed=2, radius=1, rng_seed=11)

print("Performing Edge Bundle RRT")
vis_eb_rrt  = EdgeBundleRRTAnimator( 
            start=start, goal=goal,
            goal_radius=goal_radius, 
            env = env, agent=agent, 
            edge_bundle=edge_bundle,
            max_iter = 5000,   
            # UniCycle agent util funcs      
            # isvalid_function=is_new_node_valid,
            # cost_function=get_cost,
            # random_point_function=get_random_point, 
            # reached_goal_function = agent_reached_goal,
            # translate_function=point_translate_function,
            # Mecanum agent util funcs 
            isvalid_function=Mecanum.is_new_node_valid, 
            cost_function=Mecanum.get_cost,
            random_point_function=Mecanum.get_random_point, 
            reached_goal_function = Mecanum.agent_reached_goal,
            translate_function=Mecanum.point_translate_function,
            udf_seed = 7777
           )

vis_eb_rrt.plan_path()
# path_ids, path_states, controls, timesteps = vis_eb_rrt.get_path()
# v = RRTPrinter(env, vis_eb_rrt, path_ids)
# v.print_rrt('media/edge_bundle_rrt_graph_unicycle.png')


# rrt_node_ids, states, actions, timesteps = eb_rrt.plan_path()
# v = RRTPrinter(env, eb_rrt, rrt_node_ids)
# v.print_rrt('media/edge_bundle_rrt_graph_unicycle.png')
