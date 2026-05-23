import sys
sys.path.append('./src')
import matplotlib.pyplot as plt
import networkx as nx
from kinodynamic_TI_eb_rrt import *
from printer import RRTPrinter

class KinoTIEdgeBundleRRTAnimator(KinoTIEBRRT):
    def __init__(self, * , start, goal, goal_radius, env, agent, 
                    edge_bundle,
                    use_fixed_sampling_time=True, 
                    sampling_time_step=1.0,
                    minimum_time_step=0.1,
                    max_iter=1000,
                    planning_time=np.Inf, 
                    isvalid_function,
                    cost_function,
                    reached_goal_function, 
                    random_point_function,
                    translate_function,
                    max_num_edges_per_node=1000,
                    num_skip_edges=50,
                    num_random_edges=10, 
                    eb_kd_tree,
                    get_eb_kd_tree_query,
                    kd_tree_delta_radius=0.5,
                    udf_seed=7,
                    debug_flag=False,
                    print_logs=False,
                    print_eb = True
                    ):

        super().__init__(start=start, goal=goal, goal_radius=goal_radius, env=env, agent=agent, 
                    edge_bundle=edge_bundle,
                    use_fixed_sampling_time=use_fixed_sampling_time, 
                    sampling_time_step=sampling_time_step,
                    minimum_time_step=minimum_time_step,
                    max_iter=max_iter,
                    planning_time=planning_time, 
                    isvalid_function=isvalid_function,
                    cost_function=cost_function,
                    reached_goal_function=reached_goal_function, 
                    random_point_function=random_point_function,
                    translate_function=translate_function,
                    sort_edges_function=self.sort_edges,
                    #Setting it to none because we will overload the sort_edges function in this class
                    max_num_edges_per_node=max_num_edges_per_node,
                    num_skip_edges=num_skip_edges,
                    num_random_edges=num_random_edges, 
                    eb_kd_tree=eb_kd_tree,
                    get_eb_kd_tree_query=get_eb_kd_tree_query,
                    kd_tree_delta_radius=kd_tree_delta_radius,
                    udf_seed=udf_seed,
                    debug_flag=debug_flag,
                    print_logs=print_logs,
                )
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
        self.potential_new_point = None  # potential new point

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
            # p = 50
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
            pass
        RRTPrinter.print_path_ani(self.ax, [nodes[-1]], col='blue')
        # render agent
        RRTPrinter.print_agent(self.ax, self.agent, nodes[-1].state[:2])


        if self.last_random_point is not None:
            # show the goal point if it exists
            self.ax.scatter(*self.last_random_point[0:2], color='xkcd:dusty pink', s=100, label='Random Point')
        
        if self.potential_new_point is not None:
            self.ax.plot(self.potential_new_point[0], self.potential_new_point[1], color='black', marker='.', markersize=18)

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
    
    def sort_edges(self, closest_tree_point, random_point, start_states,
                final_states, curr_edge_indices, curr_edge_mask,
                distance_array):
        """
        Sort edges according to distance to random point

        :MAINT: Overloaded to catch list of sorted eb, print info to console

        Returns:
            list(int): indices into edge bundles list, sorted by distance to random_point
                after propagating from closest_tree_point increasing 
        """
        # eb = self.edge_bundle
        # capture last random point
        self.last_random_point = random_point
        n = curr_edge_indices.shape[0]
        num_valid_edges = 0
        for i in range(n):
            edge_idx = curr_edge_indices[i]
            if curr_edge_mask[i]:
                distance_array[i] = 1e10
            else:
                potential_new_point = self.translate(closest_tree_point,
                                        start_states[edge_idx],final_states[edge_idx])
                dist = self.agent.get_distance(potential_new_point,random_point)
                distance_array[i] = dist
                num_valid_edges += 1

        sorted_indices = np.argsort(distance_array[0:n])
        new_edge_idx = curr_edge_indices[sorted_indices[0]]
        potential_new_point = self.translate(closest_tree_point,
            start_states[new_edge_idx], final_states[new_edge_idx])
        print(f"Closest tree point: {closest_tree_point}")
        print(f"Closest edge: {sorted_indices[0]} with distance: {distance_array[sorted_indices[0]]:.2f}")
        print(f"Closest edge point: {potential_new_point}")
        print(f"Best Edge from the Bundle: {final_states[new_edge_idx]}")
        self.potential_new_point = potential_new_point
        # capture sorted indices 
        self.sorted_edges = curr_edge_indices[sorted_indices[:num_valid_edges]]
        return sorted_indices[:num_valid_edges], num_valid_edges


# Example usage
from Environments import SquareEnvironment, CircularObstacle2D, RectangleObstacle2D
from Agents import UniCycle, SecondOrderCar
from kinodynamic_TI_eb_rrt import * 
from edge_bundle import EdgeBundle, GenerateEdgeBundle
from printer import *
from kd_tree_unicycle import *
from kd_tree_second_order_car import *

from mapf_env_square_agent_unicycle import get_unicycle_agent
from mapf_env_square_agent_second_order_car import get_second_order_car_agent

agent_type = 'uni'  #'uni' or 'soc'
agent_type = 'soc'  #'uni' or 'soc'

goal = (8.0, 8.0)
goal_radius = 0.5

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
obstacles = []
                    
env = SquareEnvironment(40, 40, obstacles)

# """
goal_radius = 0.25
starts = [(1.5,5.0,0.0,0.0,0.0),
        #   (8.5,5.0,0.0,0.0,0.0),
          ]
goals = [(8.5, 2.5), 
        #  (1.5, 2.5)
         ]

num_agents= len(starts)

obstacles = [
            RectangleObstacle2D(x = 5.0, y=1.75, w=3, h=3.5),
            RectangleObstacle2D(x = 5.0, y=7.5, w=3, h=5),        
            ]
obstacles = []

env = SquareEnvironment(10, 10, obstacles)
# """

if agent_type == 'uni':
    #Unicycle Kinodynamic Edge Bundle RRT Example
    agent = get_unicycle_agent(1)

    edge_bundle_file_location = 'edge_bundles/eb_unicycle_kinodynamic_TI_edges_100000.npz'
    data = np.load(edge_bundle_file_location)
    kino_TI_eb_unicycle = EdgeBundle(data, fix_num_edges=30000, use_all_edges=False)
    edge_ids = np.arange(kino_TI_eb_unicycle.num_edges, dtype=np.int64)
    thetas = kino_TI_eb_unicycle.start_states[:, 2]  # heading angle θ
    kd_tree_TI_eb_unicycle = CircularAngleIndexNumba(thetas, ids=edge_ids)

    start = np.array([5.0, 2.0, 0.0])  # x, y, theta
    kino_TI_eb_rrt_eb = kino_TI_eb_unicycle
    kino_TI_eb_rrt_kd_tree = kd_tree_TI_eb_unicycle


if agent_type == 'soc':
    #Second Order Car Kinodynamic Edge Bundle RRT Example
    agent = get_second_order_car_agent(1)

    edge_bundle_file_location = 'edge_bundles/eb_second_order_car_kinodynamic_TI_edges_100000.npz'
    data = np.load(edge_bundle_file_location)
    kino_TI_eb_SOC = EdgeBundle(data, fix_num_edges=50000, use_all_edges=False)
    edge_ids = np.arange(kino_TI_eb_SOC.num_edges, dtype=np.int64)
    speeds = kino_TI_eb_SOC.start_states[:, 3]  # v
    phis = kino_TI_eb_SOC.start_states[:, 4]   # phi
    v_scale = agent.max_speed
    phi_scale = agent.max_phi
    kd_tree_TI_eb_SOC = VPhiTree(speeds, phis, ids=edge_ids, 
                        v_scale=v_scale, phi_scale=phi_scale)   

    start = np.array([5.0, 2.0, 0.0, 0.0, 0.0])  # x, y, theta, v, phi
    start = starts[0]
    goal = goals[0]
    kino_TI_eb_rrt_eb = kino_TI_eb_SOC
    kino_TI_eb_rrt_kd_tree = kd_tree_TI_eb_SOC


print("Performing Kinodynamic Edge Bundle RRT for agent type:", agent_type)
s = np.random.randint(0, 1000)
s = 755
print("Seed: ", s)
vis_kino_eb_rrt  = KinoTIEdgeBundleRRTAnimator( 
            start=start, goal=goal,
            goal_radius=goal_radius,
            env = env, agent=agent, 
            edge_bundle = kino_TI_eb_rrt_eb,
            use_fixed_sampling_time=False,
            sampling_time_step=2.0,
            minimum_time_step=0.1,
            max_iter = 50,
            planning_time=600.0,
            isvalid_function=agent.is_new_node_valid,
            cost_function=agent.get_cost,
            random_point_function=agent.get_random_point,
            reached_goal_function = agent.agent_reached_goal,
            translate_function = agent.kd_tree_point_translate_function,
            max_num_edges_per_node=1000,
            num_skip_edges= 50,
            num_random_edges= 10,
            eb_kd_tree=kino_TI_eb_rrt_kd_tree,
            get_eb_kd_tree_query=agent.get_eb_kd_tree_query,
            kd_tree_delta_radius=0.1,
            debug_flag=True,
            print_logs=True,
            print_eb=True,
            udf_seed=s
            )

vis_kino_eb_rrt.plan_path()
path_ids, path_states, controls, timesteps = vis_kino_eb_rrt.get_path()
v = RRTPrinter(env, vis_kino_eb_rrt, path_ids)
v.print_rrt('media/animator_kino_TI_eb_rrt_second_order_car.png')


# rrt_node_ids, states, actions, timesteps = eb_rrt.plan_path()
# v = RRTPrinter(env, eb_rrt, rrt_node_ids)
# v.print_rrt('media/edge_bundle_rrt_graph_unicycle.png')
