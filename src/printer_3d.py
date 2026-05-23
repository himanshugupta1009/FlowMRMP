import os

os.environ.setdefault("MPLCONFIGDIR", "/tmp/mapf_matplotlib_cache")
os.makedirs(os.environ["MPLCONFIGDIR"], exist_ok=True)
import matplotlib

matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.animation as animation
from matplotlib.transforms import Affine2D
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import math 
import numpy as np
from rrt import TreeNode
from Environments import SquareEnvObsShape
from IPython.display import HTML

"""
This file holds several utilites for producing images and animations
for environments and simulations that don't use a high-level sim
environment 
"""

class RRTPrinter3d:
    def __init__(self, env, rrt, path_states):
        """Creates an object with several utils for 
        producing environments and simulations when 
        only one agent is present

        Args:
            env: environment object
            rrt: rrt instance that has performed a path-planning operation, can be None
            path_states (list(agent_state_type): List of states for an agent path 
        
        Note: This is a bit misleading. 
        path_states is not really a list of states, but rather a list of
        state IDs corresponding to nodes in the RRT tree.
        """
        self.env = env 
        
        self.nodes = {} 
        # if the rrt obj is not none, collect various data from it
        if rrt is not None:
            for node in rrt.tree.nodes(data=True):
                self.nodes[node[0]] = node[1].get('value')

            self.goal = rrt.goal
            self.start = rrt.start
            self.states = path_states
            self.agent = rrt.agent
            self.goal_radius = rrt.goal_radius
            print("Nodes gathered")
        else:
            print("Nodes gather skipped")

    @staticmethod
    def get_obstacle_patch(ob, obs_buffer_value=0., color='grey', ax=None):
        """Get a patch to show an obstacle in 3D

        Args:
            ob (AbstractObstacle): obstacle structure
            obs_buffer_value (float, optional): env obstacle buffer. Defaults to 0..
            color (str, optional): color for patch. Defaults to 'grey'.
            ax (Axes3D, optional): 3D axes for Poly3DCollection. Required for cuboids.

        Returns:
            Patch or Poly3DCollection: 3D obstacle representation
        """
        if ob.shape is SquareEnvObsShape.SPHERE:
            # Create a sphere visualization at the obstacle position
            # Using scatter to represent a sphere
            return (ob.x, ob.y, ob.z, ob.r + obs_buffer_value)
        elif ob.shape is SquareEnvObsShape.CUBOID:
            # Create cuboid using Poly3DCollection
            if ax is None:
                raise ValueError("Axes3D (ax) required for cuboid visualization")
            
            half_l = (ob.l + obs_buffer_value) / 2
            half_w = (ob.w + obs_buffer_value) / 2
            half_h = (ob.h + obs_buffer_value) / 2
            
            # Define the 8 vertices of the cuboid
            vertices = np.array([
                [ob.x - half_l, ob.y - half_w, ob.z - half_h],  # 0
                [ob.x + half_l, ob.y - half_w, ob.z - half_h],  # 1
                [ob.x + half_l, ob.y + half_w, ob.z - half_h],  # 2
                [ob.x - half_l, ob.y + half_w, ob.z - half_h],  # 3
                [ob.x - half_l, ob.y - half_w, ob.z + half_h],  # 4
                [ob.x + half_l, ob.y - half_w, ob.z + half_h],  # 5
                [ob.x + half_l, ob.y + half_w, ob.z + half_h],  # 6
                [ob.x - half_l, ob.y + half_w, ob.z + half_h],  # 7
            ])
            
            # Define the 6 faces using vertex indices
            faces = [
                [vertices[0], vertices[1], vertices[2], vertices[3]],  # bottom
                [vertices[4], vertices[5], vertices[6], vertices[7]],  # top
                [vertices[0], vertices[1], vertices[5], vertices[4]],  # front
                [vertices[2], vertices[3], vertices[7], vertices[6]],  # back
                [vertices[0], vertices[3], vertices[7], vertices[4]],  # left
                [vertices[1], vertices[2], vertices[6], vertices[5]],  # right
            ]
            
            poly = Poly3DCollection(faces, alpha=0.3, facecolor=color, edgecolor='darkgrey')
            return poly
        elif ob.shape is SquareEnvObsShape.CIRCLE:
            xy = (ob.x, ob.y)
            return patches.Circle(xy, ob.r + obs_buffer_value, color=color)
        elif ob.shape is SquareEnvObsShape.RECTANGLE:
            w = ob.w + obs_buffer_value
            h = ob.h + obs_buffer_value
            xy = (ob.x - w/2., ob.y - h/2.)
            return patches.Rectangle(xy, w, h, edgecolor='none', facecolor=color)

    @staticmethod
    def print_obs(ax, obs, obs_buffer_value=0.):
        """Adds obstacle representations to a 3D axis

        Args:
            ax (Axes3D): MatPlotLib 3D axis
            obs (list(AbstractObstacle)): list of obstacles to add to axis
            obs_buffer_value (float, optional): Obstacle buffer distance. Defaults to 0.

        Returns:
            list: List of plotted obstacle objects
        """
        plots = []
        # for each obstacle...
        for ob in obs:
            if ob.shape is SquareEnvObsShape.SPHERE:
                # Sphere: draw as scatter point
                x, y, z, r = RRTPrinter3d.get_obstacle_patch(ob, obs_buffer_value, color='grey', ax=ax)
                # Add buffer visualization if needed
                if obs_buffer_value > 0:
                    ax.scatter([x], [y], [z], c='xkcd:pale peach', s=100, alpha=0.3, marker='o')
                ax.scatter([x], [y], [z], c='grey', s=50, marker='o')
                plots.append((x, y, z, r))
            elif ob.shape is SquareEnvObsShape.CUBOID:
                # Cuboid: draw with Poly3DCollection
                if obs_buffer_value > 0:
                    buffer_poly = RRTPrinter3d.get_obstacle_patch(ob, obs_buffer_value, color='xkcd:pale peach', ax=ax)
                    ax.add_collection3d(buffer_poly)
                poly = RRTPrinter3d.get_obstacle_patch(ob, 0., color='grey', ax=ax)
                ax.add_collection3d(poly)
                plots.append(poly)
        return plots

    def print_path(self, ax, states, nodes, col='c'):
        """Adds an agent path segment(s) to a 3D axis

        Args:
            ax (Axes3D): MatPlotLib 3D axis
            states (list(int)): List of state ids to plot
            nodes: RRT tree nodes including all keys in states
            col (str, optional): Path color. Defaults to 'c'.
        """
        # for each state id, add the path segment for that node to 
        # the axis
        for state_id in states:
            rrtNode = nodes[state_id]
            pid = rrtNode.parent_id
            # if the parent node exists, plot path from the parent state
            if(pid != -1):
                parent_state = nodes[pid].state
                path = rrtNode.path_from_parent
                xs = [parent_state[0]] + [i[0] for i in path]
                ys = [parent_state[1]] + [i[1] for i in path]
                zs = [parent_state[2]] + [i[2] for i in path]
                ax.plot(xs, ys, zs, linestyle='-', color=col)

            # plot state
            state = rrtNode.state
            ax.plot([state[0]], [state[1]], [state[2]], color=col, marker='.', markersize=4)

    def print_tree(self, ax, nodes, col='y'):
        """Print paths to nodes in tree that are not in the 
        final agent path in 3D

        Args:
            ax (Axes3D): MatPlotLib 3D axis
            nodes: RRT tree nodes including all keys in states
            col (str, optional): Color for displayed paths. Defaults to 'y'.
        """
        for rrtNode in nodes.values():
            pid = rrtNode.parent_id
            
            # if the parent node exists, plot path from the parent state
            if(pid != -1):
                parent_state = nodes[pid].state
                path = rrtNode.path_from_parent
                xs = [parent_state[0]] + [i[0] for i in path]
                ys = [parent_state[1]] + [i[1] for i in path]
                zs = [parent_state[2]] + [i[2] for i in path]
                ax.plot(xs, ys, zs, color=col, linestyle='-')

            # plot state
            state = rrtNode.state
            ax.plot([state[0]], [state[1]], [state[2]], color=col, marker='.', markersize=2)

    @staticmethod
    def print_path_ani(ax, states, col='c'):
        """Adds an agent path segment(s) to a 3D axis used
        in generating an animation

        Args:
            ax (Axes3D): MatPlotLib 3D axis
            states (list(int)): List of state ids to plot
            col (str, optional): Path color. Defaults to 'c' (black).

        Returns:
            list of plots for each path segment 
        """
        plots = []
        if states == None: return plots
        for rrtNode in states:
            if(rrtNode.parent_id != -1):
                path = rrtNode.path_from_parent
                xs = [i[0] for i in path]
                ys = [i[1] for i in path]
                zs = [i[2] for i in path]
                plots.append(ax.plot(xs, ys, zs, color=col, linestyle='-')[0])

            # plot state
            state = rrtNode.state
            plots.append(ax.plot([state[0]], [state[1]], [state[2]], color=col, marker='.', markersize=4)[0])
        return plots

    @staticmethod
    def print_tree_ani(ax, nodes, all_nodes = None, col='y'):
        """Print paths to nodes in tree that are not in the 
        final agent path to a 3D axis used in generating an
        animation

        Args:
            ax (Axes3D): MatPlotLib 3D axis
            nodes (list(RRT Tree node): nodes to print path to
            all_nodes (dict(RRT Tree node), optional): All nodes from an RRT instance. 
                Defaults to None.
            col (str, optional): Color for displayed paths. Defaults to 'y'.

        Returns:
            list of plots for each path segment 
        """
        plots = []
        if nodes == None: return plots
        for rrtNode in nodes:            
            # print parent path if it exists 
            if(rrtNode.parent_id != -1):
                xs=[]
                ys=[]
                zs=[]
                if all_nodes is not None:
                    parent_state = all_nodes[rrtNode.parent_id].state
                    xs = [parent_state[0]]
                    ys = [parent_state[1]]
                    zs = [parent_state[2]]
                path = rrtNode.path_from_parent
                xs += [i[0] for i in path]
                ys += [i[1] for i in path]
                zs += [i[2] for i in path]
                plots.append(ax.plot(xs, ys, zs, color=col, linestyle='-')[0])

            # plot state
            state = rrtNode.state
            plots.append(ax.plot([state[0]], [state[1]], [state[2]], color=col, marker='.', markersize=2)[0])
        return plots 
    
    @staticmethod
    def print_goal(ax, goal, goal_radius, agent_id = ""):
        """Add a goal obj to the axis

        Args:
            ax (Axis): MatPlotLib axis
            goal (tuple(x, y)): Center location of goal
            goal_radius (float): radius from center of goal
            agent_id (str, optional): The agent to which this goal belongs. Defaults to "".

        Returns:
            (Patch, Annotation): Patch for the goal, the agent_id label (can be None)
        """
        xyz = (goal[0], goal[1], goal[2])
        ax.scatter([goal[0]], [goal[1]], [goal[2]], c='green', marker='o', s=100, label='Goal')
        anno = None  
        # if Agent ID exists, display that number on the goal region    
        if(agent_id != ""):
            ax.text(goal[0], goal[1], goal[2], agent_id, ha='right', va='top')
        return anno

    @staticmethod
    def print_start(ax, start, agent, pcol='black', agent_id = ""):
        """Adds a start location for an agent to the axis

        Args:
            ax (Axis): MatPlotLib axis
            start (tuple(x, y)): Start location for the agent
            agent: Agent object containing a 'radius' float field 
            pcol (str, optional): The color to use for the start 
                position patch. Defaults to 'black'.
            agent_id (str, optional): Agent's name. Defaults to "".

        Returns:
            (Patch, Annotation): Patch for the start, the agent_id label (can be None)
        """
        xyz = (start[0], start[1], start[2])
        ax.scatter([start[0]], [start[1]], [start[2]], c=pcol, marker='o', s=100, label='Start')
        anno = None
        # if it exists, use 
        if(agent_id != ""):
            ax.text(start[0], start[1], start[2], agent_id, ha='right', va='top')
            
        return anno
    
    @staticmethod
    def print_agent(ax, agent, xy, pcol='xkcd:powder blue', agent_id="", theta=None):
        """Adds an agent representation to the axis using an inscribed, optionally rotated rectangle.

        Args:
            ax (Axis): MatPlotLib axis
            agent: Agent object containing a 'radius' float field
            xy (tuple(x, y)): agent's center location
            pcol (str): Patch color for agent
            agent_id (str): Agent label
            theta (float): Rotation of the rectangle in radians (around its center).
                    defaults to None, in which case the agent will be drawn as a circle

        Returns:
            (Patch, Annotation): Patch for the agent, the agent_id label (can be None)
        """

        """
            Trasparent boxes
            remove grid

        """

        if theta is None:
            return RRTPrinter3d.print_agent_circle(ax, agent, xy, pcol, agent_id)

        r = agent.radius
        aspect = 2.
        r = agent.radius
        cx, cy = xy

        # Compute inscribed width/height
        w = 2 * r * (aspect / np.sqrt(1 + aspect**2))
        h = w / aspect

        # Rectangle is defined with its CENTER at (0, 0)
        rect = patches.Rectangle((-w/2, -h/2), w, h, facecolor=pcol, edgecolor=pcol, linewidth=0, label='Agent ' + agent_id)

        # Build transform: rotate around center, then translate
        t = Affine2D().rotate(theta).translate(cx, cy) + ax.transData
        rect.set_transform(t)

        ax.add_patch(rect)

        # -----------------------------
        # Front indicator circle
        # -----------------------------
        # Local front point at top center before rotation
        local_front = np.array([w/2.5, 0])

        # Rotate and translate into world coordinates
        rot = np.array([
            [np.cos(theta), -np.sin(theta)],
            [np.sin(theta),  np.cos(theta)]
        ])
        world_front = xy + rot @ local_front

        front_r = h * 0.2
        front_circle = patches.Circle(world_front, front_r, color='black')
        ax.add_patch(front_circle)

        # -----------------------------
        # Annotation
        # -----------------------------
        anno = None
        if agent_id:
            dx, dy = r*.8, r*.8
            anno = ax.annotate(agent_id, (cx + dx, cy + dy), ha='left', va='bottom')

        return rect, anno
            
    @staticmethod
    def print_agent_circle(ax, agent, xy, pcol='xkcd:powder blue', agent_id = ""):
        """Adds an agent representation to the axis

        :PRE: Assumes circular agents

        Args:
            ax (Axis): MatPlotLib axis
            agent: Agent object containing a 'radius' float field 
            xy (tuple(x, y)): agent's location
            pcol (str, optional): Patch color for agent. Defaults to 'xkcd:powder blue'.
            agent_id (str, optional): Agent's name. Defaults to "".

        Returns:
            (Patch, Annotation): Patch for the agent, the agent_id label (can be None)
        """
        patch = ax.add_patch(patches.Circle( xy, agent.radius, color=pcol, label='Agent ' + agent_id))
        anno = None
        if(agent_id != ""):
            anno = ax.annotate(agent_id, xy,
                        ha='center', va='center')
            
        return (patch, anno)
            
        
    def print_rrt(self, filename, print_tree=True):
        """
        Prints an agent's path and tree (based on params) through a 3D
        environment to an image file

        Args:
            filename (str): file location (including name and extension)
                to save image to 
            print_tree (bool, optional): Print the tree created by the planning
                process as well as as the path. Defaults to True.
        """
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        
        # set the image boundaries to match the environment 
        ax.set_xlim(0, self.env.size[0])
        ax.set_ylim(0, self.env.size[1])
        ax.set_zlim(0, self.env.size[2])
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')

        # add obstacles 
        self.print_obs(ax, self.env.obstacles, self.env.obstacle_buffer)

        # goal
        ax.scatter([self.goal[0]], [self.goal[1]], [self.goal[2]], c='green', marker='o', s=100, label='Goal')

        # tree if requested 
        if(print_tree):
            self.print_tree(ax, self.nodes)

        # path 
        self.print_path(ax, self.states, self.nodes)

        # start
        ax.scatter([self.start[0]], [self.start[1]], [self.start[2]], c='blue', marker='o', s=100, label='Start')

        fig.savefig(filename)
        plt.close(fig)

    def print_rrt_step_ani(self, filename, animation_speed = 2, print_tree=True):
        """
        Animates an agent's path and tree (based on params) through an
        environment to a .gif

        :MAINT: While other formats may be supported, only .gif output
            has been tested. 

        Args:
            filename (str): file location (including name and extension)
                to save animation to 
            animation_speed (int, optional): Wait time between each frame. 
                Lower number -> faster animation speed. Defaults to 2.
            print_tree (bool, optional): Print the tree created by the planning
                process as well as as the path. Defaults to True.
        """
        fig, ax = plt.subplots() 
        
        # set the image boundaries to match the environment 
        ax.set_xlim(0, self.env.size[0])
        ax.set_ylim(0, self.env.size[1])
        ax.set_aspect('equal', adjustable='box')

        self.print_obs(ax, self.env.obstacles, self.env.obstacle_buffer)

        # plot start/goal
        # stop
        ax.add_patch(patches.Circle( (self.goal[0], self.goal[1]), self.goal_radius, color='green'))
        # start
        ax.plot(self.start[0], self.start[1], 'bo', label='start') 
    
        print("Animating")

        n_nodes = len(self.nodes)

        # adds additional information for each step. Previous steps not
        # overwritten 
        def animate(n):
            plots = []
            rrtNode = self.nodes[n]
            if(print_tree):
                plots += self.print_tree_ani(ax, [rrtNode], all_nodes=self.nodes)
            if rrtNode.id in self.states:
                plots += self.print_path_ani(ax, [rrtNode])
            return plots

        ani = animation.FuncAnimation(fig, animate, interval=animation_speed, 
                                      frames=n_nodes, blit=True)
        print("Done animating, now saving")
        ani.save(filename=filename, writer="pillow")



class MultiRRTPrinter3d(RRTPrinter3d):
    def __init__(self, env, rrts, states, tcols, pcols, joint_states = False):
        """
        Creates an object with several utils for 
        producing environments and simulations when 
        multiple agents are present 

        Args:
            env: environment object
            rrts (list(RRT object)): rrt instance that has performed a path-planning 
                operation for each agent, can be None
            states (list(list(agent_state_type))): List of states for each agent for a path
            tcols (list(str)): list of colors to display each agent's tree in
            pcols (_type_): list of colors to display each agent's path in 
            joint_states (bool, optional): Set to true if printing CRRT. Defaults to False.
        """
        super().__init__(env, None, None)
        self.tcols = tcols
        self.pcols = pcols

        # curate info for later use
        self.goals = []
        self.starts = []
        self.agents = []
        self.goal_radius = []
        self.nodes = []
        if not joint_states:
            # most RRT planners
            self.states_list = states
            for rrt in rrts:
                rrt_nodes = {} 
                if(hasattr(rrt, "tree")):
                    for node in rrt.tree.nodes(data=True):
                        rrt_nodes[node[0]] = node[1].get('value')
                self.nodes.append(rrt_nodes)
                self.goals.append(rrt.goal)
                self.starts.append(rrt.start)
                self.agents.append(rrt.agent)
                self.goal_radius.append(rrt.goal_radius)
        else:
            # for CRRT, the joint states must be decomposed first
            self.agents = rrts.agents
            self.goals = rrts.goals
            self.starts = rrts.starts
            self.goal_radius = rrts.goal_radii
            self.states_list = [states for _ in range(len(self.agents))]
            self.nodes = [{} for _ in range(len(self.agents))]
            for node in rrts.tree.nodes(data=True):
                joint_node = node[1].get('value')
                joint_node_id = node[0]
                for agent_idx in range(len(self.agents)):
                    self.nodes[agent_idx][joint_node_id] = TreeNode(
                        joint_node_id,
                        joint_node.state[agent_idx],
                        joint_node.parent_id,
                        joint_node.parent_action[agent_idx],
                        joint_node.parent_action_duration,
                        joint_node.path_from_parent[agent_idx],
                        joint_node.time_elapsed,
                        joint_node.cost_so_far[agent_idx]
                    )


        print("Nodes gathered")


    def print_rrt(self, filename, print_tree=True):
        """
        Prints all agents' paths and trees (based on params) through a 3D
        environment to an image file

        Args:
            filename (str): file location (including name and extension)
                to save image to 
            print_tree (bool, optional): Print the tree created by the planning
                process as well as as the path. Defaults to True.
        """
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.set_xlim(0, self.env.size[0])
        ax.set_ylim(0, self.env.size[1])
        ax.set_zlim(0, self.env.size[2])
        
        ax.set_xticks([])
        ax.xaxis.pane.set_facecolor('white')
        ax.xaxis.pane.set_edgecolor('white')
        ax.yaxis.pane.fill = False
        ax.yaxis.pane.set_edgecolor('white')
        ax.set_yticks([])
        ax.zaxis.pane.set_edgecolor('white')
        ax.zaxis.pane.set_facecolor('grey')

        ax.set_zticks([])
        ax.grid(False)

        # Add obstacles
        self.print_obs(ax, self.env.obstacles, self.env.obstacle_buffer)

        agent_counter = 0
        for agent, goal, goal_radius, nodes, states, tcol, pcol in zip(self.agents, self.goals, self.goal_radius, self.nodes, 
                                           self.states_list, self.tcols, self.pcols):
            # goal
            ax.scatter([goal[0]], [goal[1]], [goal[2]], c='green', marker='o', s=100, label='Goal' if agent_counter == 0 else '')

            # tree 
            if print_tree:
                self.print_tree(ax, nodes, col=tcol)

            #path 
            self.print_path(ax, states, nodes, col=pcol)

            agent_counter+=1

        agent_counter = 0
        for start, agent, states, tcol, pcol in zip(self.starts, self.agents, self.states_list, self.tcols, self.pcols):
            # start
            ax.scatter([start[0]], [start[1]], [start[2]], c=pcol, marker='o', s=100, label='Start' if agent_counter == 0 else '')

            agent_counter+=1
            
        # ax.legend()
        fig.savefig(filename)
        plt.close(fig)

    @staticmethod
    def print_rrt_env(filename, env, agents, starts, goals, goal_radii, scols):
        """
        Prints all agents' starts and goals through a 3D
        environment with obstacles to an image file

        Args:
            filename (str): file location (including name and extension)
                to save image to 
        """
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.set_xlim(0, env.size[0])
        ax.set_ylim(0, env.size[1])
        ax.set_zlim(0, env.size[2])
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        
        RRTPrinter3d.print_obs(ax, env.obstacles, env.obstacle_buffer)

        for agent, goal, goal_radius in zip(agents, goals, goal_radii):
            # goal
            RRTPrinter3d.print_goal(ax, goal, goal_radius, str(agent.id))

        for start, agent, pcol in zip(starts, agents, scols):
            # start
            RRTPrinter3d.print_start(ax, start, agent, pcol, str(agent.id))
            
        # ax.legend()
        fig.savefig(filename)
        plt.close(fig)



    def print_rrt_animation(self, filename, animation_speed=10, print_tree=True):
        """
        Animates all agents' paths and trees (based on params) through an
        environment to a .gif

        :MAINT: While other formats may be supported, only .gif output
            has been tested. 

        Args:
            filename (str): file location (including name and extension)
                to save animation to 
            animation_speed (int, optional): Wait time between each frame. 
                Lower number -> faster animation speed. Defaults to 2.
            print_tree (bool, optional): Print the tree created by the planning
                process as well as as the path. Defaults to True.
        """
        # set up axis to match env size 
        fig, ax = plt.subplots() 
        ax.set_xlim(0, self.env.size[0])
        ax.set_ylim(0, self.env.size[1])

        ax.set_aspect('equal', adjustable='box')
        self.print_obs(ax, self.env.obstacles, self.env.obstacle_buffer)

        # lists to hold dicts of timestep to states (xy rep) 
        # and nodes (RRT node) for each agent at each timestep
        states_at_tstep = []
        nodes_at_tstep = []
        # maximum time for any agent's path 
        max_time = 0

        # iterate over each agent and all associated objs, curate 
        # time-based info for each 
        print("Start Gathering: ")
        agent_counter = 0
        for agent, goal, start, goal_radius, nodes, states, pcol in zip(self.agents, self.goals, self.starts, self.goal_radius, 
                                                                 self.nodes, self.states_list, self.pcols):
            print("Gathering for agent number: " + str(agent_counter))
            # plot start/goal
            # goal
            self.print_goal(ax, goal, goal_radius, str(agent_counter))
            # start
            self.print_start(ax, start, agent, pcol, str(agent_counter))

            # get state data keyed on time
            agent_states_at_tstep = {}
            for state_id in states:
                rrtNode = nodes[state_id]
                time = math.floor(rrtNode.time_elapsed)
                if(time not in agent_states_at_tstep):
                    agent_states_at_tstep[time] = []
                agent_states_at_tstep[time].append(rrtNode)
                if time > max_time:
                    max_time = time
            
            # get node data keyed on time
            agent_tree_nodes_at_tstep = {}
            for rrtNode in nodes.values():
                time = math.floor(rrtNode.time_elapsed)
                if(time not in agent_tree_nodes_at_tstep):
                    agent_tree_nodes_at_tstep[time] = []
                agent_tree_nodes_at_tstep[time].append(rrtNode)
                
            # add to lists for all agents 
            states_at_tstep.append(agent_states_at_tstep)
            nodes_at_tstep.append(agent_tree_nodes_at_tstep)
            agent_counter+=1

        print("Done gathering, now animating")

        # adds additional information for each step. Previous steps not
        # overwritten 
        def animate(n):
            plots = []
            # get info for each agent at the current timestep
            for i, states, nodes, tcol, pcol in zip(range(len(states_at_tstep)), 
                                                          states_at_tstep, nodes_at_tstep, self.tcols, self.pcols):
                if print_tree:
                    plots += self.print_tree_ani(ax, nodes.get(n), col=tcol, all_nodes=self.nodes[i])
                plots += self.print_path_ani(ax, states.get(n), pcol)
            return plots 

        ani = animation.FuncAnimation(fig, animate, interval=animation_speed, frames=max_time, blit=True)
        print("Done animating, now saving")
        ani.save(filename=filename, writer="pillow")

    def print_simulation(self, filename, animation_speed=10):
        """
        Animates all agents' paths through an environment, representing
            each agent's state at a timestep to a .gif. No paths are rendered

        :MAINT: While other formats may be supported, only .gif output
            has been tested. 
        :MAINT: This method is inferior to the highres version below. Typically, 
            you should use that instead! 

        Args:
            filename (str): file location (including name and extension)
                to save animation to 
            animation_speed (int, optional): Wait time between each frame. 
                Lower number -> faster animation speed. Defaults to 2.
        """
        fig, ax = plt.subplots() 
        ax.set_xlim(0, self.env.size[0])
        ax.set_ylim(0, self.env.size[1])
        ax.set_aspect('equal', adjustable='box')

        # get all obstacles. These patches will be reused for each frame
        printed_obs = self.print_obs(ax, self.env.obstacles, self.env.obstacle_buffer)

        # list of goal patches, populated in loop over agents.
        # will be reused at each frame. 
        printed_goals = []
        # list of a dict for each agent representing that agent's state
        # at each timestep
        states_at_tstep = []

        # get the longest time
        max_time = 0
        for states, nodes in zip(self.states_list, self.nodes):
            rrtNode = nodes[states[-1]]
            time = math.floor(rrtNode.time_elapsed)
            max_time = max(max_time, time)

        # iterate over each agent and all associated objs, curate 
        # time-based info for each 
        print("Start Gathering: ")
        agent_counter = 0
        for _, goal, goal_radius, nodes, states in zip(self.agents, self.goals, self.goal_radius, 
                                                                 self.nodes, self.states_list):
            print("Gathering for agent: " + str(agent_counter))
            # plot start/goal
            # goal
            printed_goals += [self.print_goal(ax, goal, goal_radius, str(agent_counter))]
            # start
            # printed_starts += [self.print_start(ax, rrt, pcol, str(agent_counter))]

            # get data keyed on time
            agent_states_at_tstep = {}

            # get the state of each agent at each timestep 
            # based on the state for each node
            max_agent_time = -1
            for state_id in states:
                rrtNode = nodes[state_id]
                time = math.floor(rrtNode.time_elapsed)
                state = rrtNode.state
                agent_states_at_tstep[time] = (state[0], state[1])
                if time > max_agent_time:
                    max_agent_time = time

            # make sure the agent sticks around after reaching goal
            if(max_agent_time < max_time):
                # handle if no path found for agent 
                # i.e. outside env, won't get shown 
                default_state = (-10,-10)
                if(len(states) != 0):
                    last_state = nodes[states[-1]].state
                    default_state = (last_state[0], last_state[1])
                for t in range(max_agent_time+1, max_time+1):     
                    agent_states_at_tstep[t] = default_state

            # make sure any gaps are filled in
            for t in range(max_agent_time):
                if t not in agent_states_at_tstep:
                    agent_states_at_tstep[t] = agent_states_at_tstep[t-1]
                
            
            states_at_tstep.append(agent_states_at_tstep)
            agent_counter+=1

        print("Done gathering, now animating")

        # Has to start every frame from scratch, replacing all
        # info for each frame 
        def animate(n):
            ax.clear()
            ax.set_xlim(0, self.env.size[0])
            ax.set_ylim(0, self.env.size[1])

            ax.set_aspect('equal', adjustable='box')

            # reuse goal patches 
            for (p, a) in printed_goals:
                ax.add_patch(p)
                ax.add_artist(a)
                
            for (p, a) in printed_obs:
                ax.add_patch(p)
                ax.add_artist(a)
            
            ax.set_title(f"Time {n}")

            agent_counter = 0
            plots = []
            for states, agent in zip(states_at_tstep, self.agents):
                # :MAINT: No 'blit' arg means each frame is re-written
                plots += self.print_agent(ax, agent, states[n], agent_id=str(agent_counter))
                agent_counter+=1
            
            return ax 

        ani = animation.FuncAnimation(fig, animate, interval=animation_speed, frames=max_time)
        print("Done animating, now saving")
        ani.save(filename=filename, writer="pillow")

    def print_highres_simulation(self, highres_paths, filename, step = 0.1, animation_speed=10):
        """
        Animates all agents' paths through a 3D environment based on the 'high-res path', 
        representing each agent's state at a timestep to a .gif. No paths are 
        rendered, instead the agent's position at each timestep is shown.

        :MAINT: While other formats may be supported, only .gif output
            has been tested. 

        Args:
            highres_paths (dict(int, dict(float, agent_state_type))): Dict of 
                highres paths for each agent keyed on agent ID
            filename (str): file location (including name and extension)
                to save animation to 
            step (float, optional): Timestep between each entry in the highres
                paths. Defaults to 0.1.
            animation_speed (int, optional): Wait time between each frame. 
                Lower number -> faster animation speed. Defaults to 10.
        """
        fig = plt.figure()
        plt.rcParams['animation.embed_limit'] = 100 * 1024  # 100 MB 
        ax = fig.add_subplot(111, projection='3d')

        ax.set_xlim(0, self.env.size[0])
        ax.set_ylim(0, self.env.size[1])
        ax.set_zlim(0, self.env.size[2])
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')

        # get all obstacles
        self.print_obs(ax, self.env.obstacles, self.env.obstacle_buffer)

        # determine the amount of time to scale each integer-based
        # index by (frame count in animator)
        scale_factor = 1./step
        # number of decimals to round to based on timestep
        decimals = (int)(abs(math.log10(step)))

        # get the longest time for any agent 
        max_time = 0
        for path_key in highres_paths.keys():
            time = round(list(highres_paths[path_key].keys())[-1], decimals)
            max_time = max(max_time, time)

        print("Start Gathering: ")
        agent_counter = 0
        # Pre-create scatter plots for each agent
        agent_scatters = []
        
        for goal, agent, path_key in zip(self.goals, self.agents, highres_paths):
            print("Gathering for agent: " + str(agent_counter))
            
            # plot goal
            ax.scatter([goal[0]], [goal[1]], [goal[2]], c='green', marker='o', s=100, 
                      label='Goal' if agent_counter == 0 else '')

            # Create a scatter plot for the agent
            scatter = ax.scatter([], [], [], c='blue', marker='o', s=50)
            agent_scatters.append(scatter)

            # Make sure the agent stays at its last position
            # until all other agents' paths have been completely rendered
            path = highres_paths[path_key]
            last_agent_time = list(path.keys())[-1]
            last_agent_state = path[last_agent_time]
            for new_step in np.arange(last_agent_time+step, max_time+step, step):
                path[round(new_step, decimals)] = last_agent_state
            agent_counter+=1

        print("Done gathering, now animating")

        def animate(n):
            timestep = round(n/scale_factor, decimals)
            ax.set_title(f"Time {timestep}")

            agent_counter = 0
            for path_key, agent, scatter in zip(highres_paths, self.agents, agent_scatters):
                state = highres_paths[path_key][timestep]
                x, y, z = state[0], state[1], state[2]
                scatter._offsets3d = ([x], [y], [z])
                agent_counter+=1
            
            return agent_scatters

        # :MAINT: Using blit=True for efficiency with pre-created artists
        ani = animation.FuncAnimation(fig, animate, interval=animation_speed, 
                                     frames=(int)(max_time*scale_factor), blit=False, )
        print("Done animating, now saving")
        html_string = HTML(ani.to_jshtml())
        with open(filename.replace('.gif', '.html'), 'w') as f:
            f.write(html_string.data)
        ani.save(filename=filename, writer="pillow")
        
        


"""
from printer import *
v = RRTPrinter(env, rrt, states)
v.print_rrt('media/rrt_graph.png')


"""
