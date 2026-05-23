import pybullet as p
import pybullet_data

from utils import euclidean_distance

import numpy as np
import quaternion
from itertools import accumulate
from enum import Enum
import bisect
import sys
sys.path.insert(0, '.')
sys.path.insert(0, '../')
# print(sys.path)
import importlib  
pybullet_utils = importlib.import_module("pybullet-planning.pybullet_tools.utils")

from Environments import AbstractObstacle

TURTLEBOT_NONHOLO_URDF = pybullet_utils.join_paths(pybullet_utils.MODEL_DIRECTORY, 'turtlebot/turtlebot.urdf')


"""
:MAINT: A lot of this design is quite dodgy, but is the best that can be done 
to get PyBullet to work within the existing framework. 
"""

"""
COLOR UTILS:
"""
def hex_to_rgba(hex_color):
    """
    Converts a string hex to a int list rgba

    Args:
        hex_color (str): hex color strin

    Returns:
        list(int): RGBa for the hex value
    """
    col_val = [int(hex_color[i:i+2], 16)/(2**8 - 1) for i in (0, 2, 4)] + [1]
    return col_val

def bot_to_hex(hex_color, bot):
    """
    Converts an entire PyBullet agent to a hex color

    Args:
        hex_color (str): hex color string
        bot (int): pb id for a multi-joint object that HAS BEEN LOADED
    """
    col_val = hex_to_rgba(hex_color)
    # change the color for each joint
    for j in range (-1, p.getNumJoints(bot)):
        p.changeVisualShape(bot, j, rgbaColor=col_val) 

class PyBulletEnv():

    # list of possible agent colors 
    agent_colors = ["c7fdb5", "ceb301", "ff796c", "8e82fe", "6c3461", "98eff9"]

    def __init__(self, length, depth, obs, use_gui=True, speed=240., draw_borders=True):
        """
        Get a new PyBullet Env 

        Args:
            length (float): environment 'x' size
            depth (float): environment 'y' size
            obs (list(AbstractObstacle)): List of PyBullet obstacles to add to env
            use_gui (bool, optional): Display the environment in a GUI window. Defaults to True.
            speed (_type_, optional): Environment Hz (larger then number, slower the simulation runs). Defaults to 240.0.
            draw_borders (bool, optional): Draw the environment borders, useful to turn off
                when running headless or when generating edge bundles. Defaults to True.

        Raises:
            Exception: Unknown obstacle type in obs
        """        
        # length is x, depth is y
        self.size = (length, depth)

        # these are unused for PyBullet sims, but set for consistency 
        self.boundary_buffer = 0.1
        self.obstacle_buffer = 0.5

        self.speed = speed
        self.p_inst = pybullet_utils.connect(use_gui=use_gui)    
        # TODO: figure out how to set camera to a better spot
        # if (use_gui):
        pybullet_utils.set_camera_pose(camera_point=[1, -1.5, 1], target_point=pybullet_utils.unit_point()) # Sets the camera's position
        # set_camera_pose(camera_point=[length/2., breadth/2., (length + breadth)/2], 
        #                 target_point=(length/2., breadth/2., 0)) # Sets the camera's position

        self.agents = []
        self.obstacles = []

        with pybullet_utils.LockRenderer(): # Temporarily prevents the renderer from updating for improved loading efficiency
            with pybullet_utils.HideOutput(): # Temporarily suppresses pybullet output
                # Load env static objects (obstacles, boundaries)
                p.setAdditionalSearchPath(pybullet_data.getDataPath())
                self.plane = p.loadURDF("plane.urdf")

                # boarder fence, which is for reference ONLY (boundary enforced by math with agent pos) and therefore:
                # a) collisions are disabled
                # b) borders are overestimated 
                if draw_borders:
                    boundary_width = 0.25 # width of each individual boundary object
                    overestimation_factor = 1. # overestimation to make sure sims don't look weird
                    boundary_height = 0.2 # height of boundary in 'z' direction
                    left_center = [-(0.5 * boundary_width + overestimation_factor), depth/2., boundary_height/2. ]
                    right_center = [length + (0.5 * boundary_width + overestimation_factor), depth/2., boundary_height/2. ]
                    bottom_center = [length/2., -(0.5 * boundary_width + overestimation_factor), boundary_height/2.]
                    top_center = [length/2., depth + (0.5 * boundary_width + overestimation_factor), boundary_height/2.]

                    # set left fence
                    left_fence_id = pybullet_utils.create_box(w=boundary_width, l=(depth + 2 * (boundary_width + overestimation_factor)), 
                                                            h=boundary_height, color=pybullet_utils.BLACK, collision=False)
                    pybullet_utils.set_point(left_fence_id, left_center)
                    pybullet_utils.set_static(left_fence_id)
                    # set right fence
                    right_fence_id = pybullet_utils.create_box(w=boundary_width, l=(depth + 2 * (boundary_width + overestimation_factor)), 
                                                            h=boundary_height, color=pybullet_utils.BLACK, collision=False)
                    pybullet_utils.set_point(right_fence_id, right_center)
                    pybullet_utils.set_static(right_fence_id)
                    # set bottom fence
                    bottom_fence_id = pybullet_utils.create_box(w=(length + 2 * (boundary_width + overestimation_factor)), l=boundary_width, 
                                                            h=boundary_height, color=pybullet_utils.BLACK, collision=False)
                    pybullet_utils.set_point(bottom_fence_id, bottom_center)
                    pybullet_utils.set_static(bottom_fence_id)
                    # set top fence
                    top_fence_id = pybullet_utils.create_box(w=(length + 2 * (boundary_width + overestimation_factor)), l=boundary_width, 
                                                            h=boundary_height, color=pybullet_utils.BLACK, collision=False)
                    pybullet_utils.set_point(top_fence_id, top_center)
                    pybullet_utils.set_static(top_fence_id)

                # load static (non-agent) obs
                for given_ob in obs:
                    ob_pb_id = -9001
                    if(given_ob.type is ObsShape.CIRCLE):
                        ob_pb_id = pybullet_utils.create_cylinder(given_ob.r, given_ob.h, color=given_ob.color)
                    elif (given_ob.type is ObsShape.RECTANGLE) :
                        ob_pb_id = pybullet_utils.create_box(w = given_ob.w, l=given_ob.l, h=given_ob.h, color=given_ob.color)
                    else:
                        raise Exception("Only circle and square obs supported on load!")
                    pybullet_utils.set_point(ob_pb_id, [given_ob.x, given_ob.y, given_ob.h / 2.])
                    pybullet_utils.set_static(ob_pb_id)
                    given_ob.pb_id = ob_pb_id
                    self.obstacles.append(given_ob)
                # sim housekeeping 
                p.setRealTimeSimulation(0)
                p.setTimeStep(1. / speed)
                p.setGravity(0,0,-10)
    
    def add_agent(self, agent, start = [-10,-10,0.2], goal=None):
        """
        Adds an agent to the env

        Args:
            agent: Pybullet agent obj
            start: start position for agent. Also where to send the agent back to if not
                    actively being used 
            goal: A tuple of ((goal x, goal y, ...), goal r) to be sent to add_goal
        """
        with pybullet_utils.LockRenderer(): # Temporarily prevents the renderer from updating for improved loading efficiency
            with pybullet_utils.HideOutput(): # Temporarily suppresses pybullet output
                # load the agent model
                new_agent_pb_id = p.loadURDF(agent.urdf, start)

                # set agent color if id in list of colors 
                if agent.id in range(0, len(PyBulletEnv.agent_colors)):
                    bot_to_hex(PyBulletEnv.agent_colors[agent.id], new_agent_pb_id)

                self.agents.append(agent)
                
                # set agent pybullet env data
                agent.pb_id = new_agent_pb_id
                agent.env = self
                # set the reset state  
                agent.out_of_bounds = ((start[0], start[1], start[2]), (0, 0, 0, 1.0))
                
        # if goal is populated, add a goal to the env too
        if goal is not None:
            self.add_goal(goal[0], goal[1], agent.id)

    def reset_agent(self, agent):
        """
        Sets an agent back to its start pos

        Args:
            agent: Pybullet agent obj
        """
        agent.set_agent_oob()
    
    def add_obstacle(self, obs):
        """
        Add a dynamic obstacle to the environment 

        Args:
            obs (AbstractObstacle): dynamic obstacle to add

        Raises:
            Exception: when a non-dynamic agent is passed in 
        """
        self.obstacles.append(obs)
        if obs.type is ObsShape.MOVING:
            if obs.agent in self.agents:
                self.agents.remove(obs.agent)
        else:
            raise Exception("Only moving obstacles can be added after instantiation for now!")
                    

    def add_goal(self, goal, goal_radius, agent_id = -1):
        """
        Sets goal for visualization ONLY
        goals are set to no-contact, meaning that agents can travel THROUGH them. 
        This allows the agent to reach the goal, but also means that we cannot use
        a contact sensor to detect that an agent has reached the goal. We use 
        math with the agent's position and the goal's geometry instead.

        Args:
            goal (tuple(float, float)): x,y position for goal center
            goal_radius (float): goal radius
            agent_id (int, optional): Agent's unique ID. Defaults to -1.
        """
        with pybullet_utils.LockRenderer(): # Temporarily prevents the renderer from updating for improved loading efficiency
            with pybullet_utils.HideOutput(): # Temporarily suppresses pybullet output
                color = pybullet_utils.GREEN
                # if the agent_id is within the color list's size, use that
                # color for the goal 
                if agent_id in range(0, len(PyBulletEnv.agent_colors)):
                    color = hex_to_rgba(PyBulletEnv.agent_colors[agent_id])
                goal_pb_id = pybullet_utils.create_cylinder(goal_radius, 0.1, color=color, collision=False)
                pybullet_utils.set_point(goal_pb_id, [goal[0], goal[1], 0.05])
                pybullet_utils.set_static(goal_pb_id)

    def is_collision(self, first_agent, first_agent_position, second_agent, second_agent_position):
        """
        Checks for a collision between two agents in this environment 

        :PRE: ASSUMES BOTH AGENTS ARE ALREADY IN THIS ENV and that 
        they are NOT the SAME. 

        Args:
            first_agent: A PyBullet agent
            first_agent_position (agent_state_type): the first agent's position
            second_agent: A PyBullet Agent
            second_agent_position (agent_state_type): the second agent's position 

        Returns:
            bool: True if there is a collision, False else
        """
        first_agent.set_agent(first_agent_position)
        second_agent.set_agent(second_agent_position)
        return pybullet_utils.pairwise_collision(first_agent.pb_id, second_agent.pb_id)

    def replay_path(self, states_dicts):
        """
        Relays the path with dynamic user controls 

        (->): move sim forward in time
        (<-): move sim backward in time
        (p): play/pause
        (r): set time to zero (restart)
        (ENTER): close window gracefully 

        Args:
            states_dicts (dict(float, agent_state_type)): 'high-res' paths for each agent 
        """
        # get lists of the keys, necessary for sorting later
        state_keys = [list(states_dicts[i].keys()) for i in range(len(states_dicts))] 

        # set each agent at the start of their paths
        for i in range(len(states_dicts)):
            self.agents[i].set_agent(states_dicts[i][state_keys[i][0]])

        dt = 1.0/self.speed
        t = 0. 
        # flag for if sim is in 'play' mode, where time increases passively 
        time_is_spinning = False
        # flag for if sim should stop
        quitter = False

        # get the max time over all agent paths
        max_t = 0
        for state_keys_item in state_keys:
            curr_t = state_keys_item[-1]
            if curr_t > max_t:
                max_t = curr_t

        p.setTimeStep(dt)
        while not quitter:
            # set the agents in the env at dt
            for i, agent, states_dict, state_keys_item in zip(
                range(len(self.agents)), self.agents, states_dicts, state_keys):
                agent.replay_path(t, states_dict, state_keys_item)

            # capture keyboard input
            keys = p.getKeyboardEvents()
            for k,v in keys.items():
                if (k == p.B3G_RIGHT_ARROW):
                    # move time forward
                    t += dt
                    if t > max_t:
                        t = max_t
                if (k == p.B3G_LEFT_ARROW):
                    # move time backward
                    t -= dt
                    if t < 0:
                        t = 0
                # enter to quit
                if k == 65309 and (v&p.KEY_WAS_TRIGGERED):
                    quitter = True
                # (P)lay/(P)ause  
                if k == 112 and (v&p.KEY_WAS_RELEASED):
                    time_is_spinning = not time_is_spinning
                # (R)estart
                if k == 114 and (v&p.KEY_WAS_RELEASED):
                    t = 0
                    continue
                
            if time_is_spinning:
                t+=dt 
                if t > max_t:
                    t = max_t
                    # pause simulation if the last time is reached 
                    time_is_spinning = False
            p.stepSimulation()

        pybullet_utils.disconnect()

    def replay_path_dynamics(self, states, controls, timesteps, check_cols = False):
        """
        Replays agents paths using dynamics between each node state 

        (->): move sim forward in time one second
        (p): play/pause
        (r): set time to zero (restart)
        (ENTER): close window gracefully 

        Unfortunately, stepping backwards in time is not possible with this replay 

        Args:
            states (list(list(agent_state_type))): List of each state for a path for each agent
            controls (list(list(agent_control_type))): List of each control input for a path for each agent
            timesteps (list(list(float))): List of each timestep (non-aggregate) for a path for each agent
            check_cols (bool, optional): Set to true to check collisions at every step . Defaults to False.
        """
        state_keys =  [ [0.] + list(accumulate(timestep)) for timestep in timesteps]
        for i in range(len(states)):
            self.agents[i].set_agent(states[i][0])

        dt = 1.0/self.speed
        t = 0. 
        time_is_spinning = False
        quitter = False

        max_t = 0
        for agent_state_keys in state_keys:
            if agent_state_keys[-1] > max_t:
                max_t = agent_state_keys[-1]

        last_index = [0 for _ in states]
        spin_to = -1 

        p.setTimeStep(dt)
        while not quitter:
            for i, agent, state, control, steps, in zip(
                range(len(self.agents)), self.agents, states, controls, state_keys):
                last_index[i] = agent.replay_path_dynamics(t, state, control, steps, last_index[i])

            keys = p.getKeyboardEvents()
            for k,v in keys.items():
                # enter to quit
                if k == 65309 and (v&p.KEY_WAS_TRIGGERED):
                    quitter = True 
                # (P)lay/(P)ause 
                if k == 112 and (v&p.KEY_WAS_RELEASED):
                    time_is_spinning = not time_is_spinning
                # One more second
                if (k == p.B3G_RIGHT_ARROW) and (v&p.KEY_WAS_TRIGGERED):
                    spin_to = t + 1
                    time_is_spinning = True
                # (R)estart
                if k == 114 and (v&p.KEY_WAS_RELEASED):
                    t = 0.
                    last_index = [0 for _ in states]
                    continue

            if spin_to != -1 and t >= spin_to:
                time_is_spinning = False 
                spin_to = -1

            if time_is_spinning:
                t+=dt 
                if t > max_t:
                    t = max_t
                    time_is_spinning = False
                # print(t)
                p.stepSimulation()
                if check_cols:
                    col_string = self.check_collisions()
                    if col_string != "":
                        print(col_string)
        
        pybullet_utils.disconnect()

    def check_collisions(self, get_col_string = False):  
        """
        Check collisions between all agents and between any agent and any obstacle 

        Args:
            get_col_string (bool, optional): Gather a description of the collision. 
                Defaults to False.

        Returns:
            string: Collision description string (could be empty)
        """
        col_string = ""      

        # Check each agent...
        for agent_id_1 in range(len(self.agents)):
            agent_1 = self.agents[agent_id_1]
            # ...against every other agent...
            for agent_id_2 in range(agent_id_1+1, len(self.agents)):
                agent_2 = self.agents[agent_id_2]
                if pybullet_utils.pairwise_collision(agent_1.pb_id, agent_2.pb_id):
                    # set collision flags for both involved agents. These will be used later
                    # when the planner asks each agent if a new path is valid.  
                    agent_1.collided = True 
                    agent_2.collided = True
                    if get_col_string:
                        col_string += "Agent " + str(agent_1.id) + " hit agent " + str(agent_2.id) +".\n"
        
            # ...and against every obstacle
            for obs in self.obstacles:
                if pybullet_utils.pairwise_collision(agent_1.pb_id, obs.pb_id):
                    agent_1.collided = True
                    if get_col_string:
                        col_string += "Agent " + str(agent_1.id) + " hit obstacle."
        
        return col_string

    def step_pb(self):
        p.stepSimulation()


    def step_environment(self, states, controls, dt, start_t, num_steps=10):
        """
        Steps everything that moves in the environment, be it multiple agents 
        and/or moving obstacles 

        Args:
            states: list of states, one for each active agent
            controls: list of controls, one for each active agent
            dt: time to move environment 
            start_t: global start time for the step
            num_steps: number of steps to record in env
        """
        # set each agent at start
        for state, agent in zip(states, self.agents):
            agent.set_agent(state)
            agent.reset_path()
        
        p.stepSimulation()

        t = start_t
        pb_dt = 1./self.speed
        # using 'stepper' and 'record_steps' as well as t and pb_dt mitigates float errors
        stepper = 0
        num_pb_steps = dt*self.speed
        record_steps = int(num_pb_steps/num_steps)
        # tell agents to record their state
        record_flag = False
        # while there are still pybullet steps left
        while stepper <= num_pb_steps:
            if stepper > 0 and stepper % record_steps == 0:
                record_flag = True 
            else:
                record_flag = False 
            
            # move all agents
            for control, agent in zip(controls, self.agents):
                agent.move_agent(control, record_flag)

            # move all moving obstacles
            for obs in self.obstacles:
                if obs.type is ObsShape.MOVING:
                    obs.agent.set_agent(obs.positionAtTime(t))

            stepper+=1
            t += pb_dt
            p.stepSimulation()
            self.check_collisions()

                
def get_dtype_from_input(t):
    data_type = [(f'field_{i}', type(element)) for i, element in enumerate(t)]
    return np.dtype(data_type)

class PyBulletTurtle:
    
    def __init__(self, speed, agent_id = 1, rng_seed=11):
        """
        Get a new PyBullet turtlebot (differential drive) agent 

        Args:
            speed (float): Max control value for each wheel 
            agent_id (int, optional): Hopefully unique agent id. Defaults to 1.
            rng_seed (int, optional): Random number generator seed. Defaults to 11.
        """
        self.id = agent_id
        self.state_length = 3
        self.speed = speed
        self.rng = np.random.default_rng(rng_seed)
        self.state_datatype = get_dtype_from_input(((7.0, 1.0, 0.1), (0., 0., 0., 1.0)))
        self.static_control = (0,0) # control for no movement
        # will instantiate when added to env
        self.pb_id = None 
        self.env = None # circular dependencies built the nation
        self.static_vel = ((0.0, 0.0, 0.0), (0.0, 0.0, 0.0)) # velocity at rest
        self.collided = False # collision flag
        self.urdf = TURTLEBOT_NONHOLO_URDF # file that contains info on agent
        self.out_of_bounds = ((-10, -10, 0.1), (0, 0, 0, 1.0)) # state to put agent back to when we're not using it
        # a path to a new state, to be retrieved if the env was stepped in concert by a planner
        # via is_node_valid
        self.path = None
    
    
    def get_random_action(self,udf_rng):
        """
        Gets a random action within speed bounds

        udf_rng: a random number generator to use for new 
            random control inputs

        return: a 2-tuple of random wheel velocities
        """
        vl = udf_rng.uniform(-self.speed, self.speed)
        vr = udf_rng.uniform(-self.speed, self.speed)
        return (vl, vr)
    
    def get_agent_state(self):
        """
        Gets the agent's current state

        Returns:
            agent_state_type: agent state right now
        """
        return p.getBasePositionAndOrientation(self.pb_id)
    
    def check_all_collisions(self):
        """
        Check collisions for this against all agents and obstacles in 
        its environment 

        Returns:
            boolean: True if agent is in collision state, false else
        """                
        # if the agent is hitting an obstacle, return True
        # If it isn't, need to check against the agents in the 
        # env as well. 
        if self.check_obs_collisions(): return True
        
        # Check if the agent is in contact with another agent 
        for agent in self.env.agents:
            if agent is not self:
                if pybullet_utils.pairwise_collision(self.pb_id, agent.pb_id):
                    # if it is in contact with another agent, that's enough, return
                    # true
                    return True

        # If we got here the agent is in the clear
        # if(collision): print("Hit something!")
        return False
    
    def check_obs_collisions(self): 
        """
        Check collisions against obstacles in environment

        Returns:
            bool: True if this is hitting an obstacle, false else
        """    
        # Check against each obstacle           
        for obs in self.env.obstacles:
            if pybullet_utils.pairwise_collision(self.pb_id, obs.pb_id):
                # if it is in contact with any object, that's enough, return
                # true
                return True

        # if(collision): print("Hit something!")
        return False
    
    def move_agent(self, control, save_path = False):
        """
        Set control inputs to agent wheels 

        Args:
            control (tuple(float, float)): Wheel target velocities for (left, right) wheel
            save_path (bool, optional): If true, add the current state to self.path. Defaults to False.
        """
        # NOTE: Change the joint ids to 3 and 4 for holonomic turtle, 0,1 for regular 
        p.setJointMotorControl2(self.pb_id,0,p.VELOCITY_CONTROL,targetVelocity=control[0],force=1000)
        p.setJointMotorControl2(self.pb_id,1,p.VELOCITY_CONTROL,targetVelocity=control[1],force=1000)
        if save_path:
            self.path.append(self.get_agent_state())

    def set_agent(self, state):
        """
        Put the agent at the provided state

        Args:
            state (agent_state_type): state to place agent
        """
        p.resetBasePositionAndOrientation(self.pb_id, *state)
        # set velocity to stabilize, think Quicksilver 
        p.resetBaseVelocity(self.pb_id, self.static_vel)
        self.move_agent(self.static_control)

    def set_agent_oob(self):
        """
        Move the agent to its 'out-of-the-way' state. 
        """
        self.set_agent(self.out_of_bounds)
        
    def reset_path(self):
        """
        Set the path to empty after its been retrieved
        """
        self.path = [] #np.empty(num_steps)

    def step_agent(self, control, save_path):
        """
        Apply a control input to the agent, saving the path if necessary

        Args:
            control (tuple(float, float)): Wheel target velocities for (left, right) wheel
            save_path (bool, optional): If true, add the current state to self.path. Defaults to False.
        """
        if save_path: self.path.append(self.get_agent_state())
        self.move_agent(control)

    def get_next_state(self, state, control, dt, num_steps=10):
        """
        Get the next state

        Args:
            state (agent_state_type): start state to move from
            control (tuple(float, float)): Wheel target velocities for (left, right) wheel
            dt (float): time over which to propagate 
            num_steps (int, optional): Number of states to return in the path. Defaults to 10.

        Returns:
            (agent_state_type, list(agent_state_type)): (agent's new state after path, 
                agent's path from the start state to the new state)
        """
        # if this agent has already been propagated with the whole env in step_env, 
        # return the resulting path from that 
        if self.path is not None:
            return self.get_agent_state(), self.path

        # set the agent at the start state
        self.set_agent(state)
        p.stepSimulation()

        # number of PyBullet steps taken
        stepper = 0
        # number of PyBullet steps to take
        num_pb_steps = dt*self.env.speed
        # number of PB states after which a save to the path is required
        record_steps = int(num_pb_steps/num_steps)
        # number of steps already recorded to path 
        record_steps_counter = 0
        path = [] #np.empty(num_steps)
        while stepper <= num_pb_steps:
            # While there are still PB steps to take
            if stepper > 0 and stepper % record_steps == 0:
                # if we're due to add a new state to the path
                path.append(self.get_agent_state())
                record_steps_counter+=1
            
            # set the control inputs
            self.move_agent(control)
            p.stepSimulation()
            # check for collisions while still remembering if
            # the agent has already collided earlier in the path 
            self.collided = self.collided or self.check_obs_collisions()

            # iterate number of PB states taken 
            stepper+=1

        return self.get_agent_state(), path

    # used externally, other agents have to implement this!!!
    def get_distance(self, in_state1, state2):
        """
        Get the distance between a Turtle state and a 2-tuple

        Args:
            in_state1 (agent_state_type): an agent state
            state2 (tuple(float, float)): an (x, y) tuple  

        Returns:
            _type_: _description_
        """
        # convert the turtle state to a tuple(x,y)
        state1 = PyBulletTurtle.state_to_euclid(in_state1)
        d = euclidean_distance((state1[0], state1[1]), (state2[0], state2[1]))
        return d

    # used externally for kcbs, other agents have to implement this!!!
    def get_collided(self):
        """
        Return true if this agent has hit something since last check

        Returns:
            bool: True if agent has hit something, False else
        """
        collided = self.collided 
        # reset flag for next check
        self.collided = False
        return collided      
    
    # replay method for vis
    def replay_path(self, t, state_dict, state_keys):
        """
        Called by env.replay_path, sets the agent at a 
        location from its 'high-res' path 

        :PRE: state_keys should be SORTED (can take for granted
            if the state_dict was generated by a planner's 'high-res
            path' function).

        Args:
            t (float): current time
            state_dict (dict(float, agent_state_type)): Dict from timesteps to agent's state type
            state_keys (_type_): list of the keys to state dict
                :MAINT: Python dictionary keys() are NOT lists! However, 
                    we need a list for the bisect utils used here. So 
                    convert once in the caller and use for whole replay. 

        Raises:
            Exception: _description_
        """
        pos = None        
        if (t<0): raise Exception("Time cannot be negative")
        # find index at which the current time would be inserted 
        # into the state keys
        idx = bisect.bisect_right(state_keys, t) - 1
        # if the index is beyond the length of the path, 
        # the requested time is beyond this agent's path
        if idx >= len(state_keys) - 1: 
            # Return the last state
            pos = state_dict[state_keys[-1]]
        # If the requested time is not less than zero or greater
        # than the last state, it must be between two other states.
        # Find the closest in time between the bracketing states 
        # and use it. 
        elif t - state_keys[idx] < state_keys[idx+1] - t:
            pos = state_dict[state_keys[idx]]
        else:
            pos = state_dict[state_keys[idx+1]]
        # put the agent at the position requested by t
        self.set_agent(pos)
    
    # replay method for vis w/ actual dynamics
    def replay_path_dynamics(self, t, states, controls, timesteps, last_index):
        """Called by 

        Args:
            states (list(agent_state_type))): List of each state for a path for this agent
            controls (list(agent_control_type))): List of each control input for a path for this agent
            timesteps (list(float))): List of each timestep (aggregate) for a path for this agent
            last_index (_type_): _description_

        Raises:
            Exception: Time cannot be negative

        Returns:
            int: last index into states/controls/timesteps used. In subsequent calls this value
                will be used as a reference on where to look in the lists 
        """
        if (t<0): raise Exception("Time cannot be negative")
        
        index = last_index 
        # If time has gone beyond the duration of this path, 
        # return the final state
        if last_index >= len(states) - 1:
            self.set_agent(states[-1])
            return last_index
        # If time has gone beyond the last_index's time, 
        # move up to the next index
        elif t >= timesteps[last_index+1]:
            index += 1
            self.set_agent(states[index])
            # if there is a new control to use at the 
            # new index, use it
            if index > len(controls) - 1:
                return index 
        
        # set control values if needed. 
        self.move_agent(controls[index])
        return index
        
        
    # used externally, other agents have to implement this!!!
    @staticmethod
    def is_state_valid(env, agent, state, t = 0, dt = 0):
        """
        Make sure a Turtle agent is within the environment at a state

        Args:
            env: Environment object 
            agent: Turtle agent object, Not used, left to maintain compatibility 
            state (Turtle state tuple): State for which to check validity
            t (int, optional): current time. Not used, left to maintain 
                compatibility. Defaults to 0.
            dt (int, optional): Timestep. Not used, left to maintain 
                compatibility. Defaults to 0.

        Returns:
            bool: True if state is valid, False else
        """
        # check if agent state x component is within the env
        if(state[0] < ( env.boundary_buffer) 
            or state[0] > (env.size[0]-env.boundary_buffer)):
            return False
        # check if agent state y component is within the env
        if(state[1] < (env.boundary_buffer)
                or state[1] > (env.size[1]-env.boundary_buffer)):
            return False
        return True

    # used externally, other agents have to implement this!!!
    @staticmethod
    def is_new_node_valid(env, agent, path_to_new_state, t = 0, dt = 0, 
                          check_goal=False):
        """
        Checks if a new state and the path to that new state is valid for 
        an environment.

        :PRE: Agent has been propagated along this path, setting self.collided
            as necessary. 

        Args:
            env: PyBullet environment 
            agent (PyBulletTurtle): Turtle agent object,
            path_to_new_state (list[Turtle state tuple]): Path taken by the agent 
                from the parent state, non-inclusive of the parent state but inclusive
                of the final state 
            t (int, optional): current time. Not used, left to maintain 
                compatibility. Defaults to 0.
            dt (int, optional): Timestep. Not used, left to maintain 
                compatibility. Defaults to 0.
            check_goal (bool, optional) Check if this state is valid as a final state
                for the agent. Defaults to False

        Returns:
            bool: True if state is valid (i.e. the agent didn't hit anything or the boundary), 
                False else
        """
        # If the agent hit something on the path, return false
        if agent.get_collided():
            return False
        
        # get last state in path as tuple
        state = [0,0]
        state[0] = path_to_new_state[-1][0][0]
        state[1] = path_to_new_state[-1][0][1]

        is_new_state_valid =  PyBulletTurtle.is_state_valid(env, agent, state)

        if check_goal:
            # check against moving obs
            # in the env.
            print("GOAL CHECK MECANUM") 
            path_timestep = dt/len(path_to_new_state)
            state = path_to_new_state[-1]
            current_time = t+dt 
            for obs in env.obstacles:
                if obs.type is ObsShape.MOVING:
                    # if the obstacle is a moving obstacle, it 
                    # is an agent with a planned path. We must 
                    # check the rest of that path against this 
                    # agent's position to make sure that these 
                    # already-planned agents will not hit it 
                    # as they finish up their paths
                    t_max = obs.max_time
                    t = current_time
                    while t <= t_max:
                        obs_pos = obs.positionAtTime(t)
                        if env.is_collision(agent, state, 
                                            obs.agent, obs_pos):
                            found_goal = False
                            break 
                        t += path_timestep
                # break out of loop over obstacles 
                if not found_goal:
                    break

        return is_new_state_valid

    # used externally, other agents have to implement this!!!
    @staticmethod
    def get_random_point(env, agent, rng):
        """
        Gets a random point for a Turtle agent in an environment 


        Args:
            env: Environment object 
            agent: PyBullet Turtle agent object
            rng (numpy.random): random number generator

        Returns:
            tuple(x, y): new valid point for the agent in the environment  
        """        
        x = rng.uniform(0, env.size[0])
        y = rng.uniform(0, env.size[1])
        point = (x, y)

        # make sure the new point is in-bounds. If not, try to get a 
        # new state 
        while not PyBulletTurtle.is_state_valid(env, agent, point):
            x = rng.uniform(0, env.size[0])
            y = rng.uniform(0, env.size[1])
            # theta = rng.uniform(0, 2 * np.pi)
            # p = (x, y, theta)
            point = (x, y)
        return point 
    
    # used externally, other agents have to implement this!!!
    @staticmethod
    def get_cost(env, agent, parent_state, a, t, edge): 
        """
        Gets a cost estimate for what a Turtle agent incurs to traverse 
        a path

        :MAINT: Arg list with unused elements to maintain 
        consistency with other agents 

        Args:
            env: Environment object 
            agent: PyBulletTurtle agent object
            parent_state (agent_state_type): Start point for path 
                NOT USED
            a (agent_control_type): control input that generated
                path NOT USED
            t (float): Time over which path was propagated NOT USED
            edge (list(agent_state_type)): new path over which 
                to generate cost

        Returns:
            float: approximate path cost
        """ 
        path_dist_approx = 0
        # start state is not included in path 
        last_state = PyBulletTurtle.state_to_euclid(parent_state)
        for path_state in edge:
            path_state_e = PyBulletTurtle.state_to_euclid(path_state)
            path_dist_approx += euclidean_distance((path_state_e[0], path_state_e[1]), 
                                                  (last_state[0], last_state[1]))
        return path_dist_approx
        
    # used externally, other agents have to implement this!!!
    @staticmethod
    def agent_reached_goal(raw_state, goal, goal_radius, agent, env=None, 
                           current_time=None, path_timestep = None):
        """
        Determines whether the agent has reached the goal region

        :MAINT: Arg list with unused elements to maintain 
        consistency with other agents 

        Args:
            state (agent_state_type): agent state
            goal (tuple(float, float)): goal center
            goal_radius (float): goal region radius
            agent: Turtle agent object
            env (optional): Environment, will be checked for 
                already-planned agents as moving obstacles  
            current_time (float, optional): current time for 
                the state argument, must be provided if env 
                is not None. 
            path_timestep (float, optional): time discretation 
                of already-planned paths for moving obstacles 
                in env. Must be provided if env is not None 
                
        Returns:
            bool: True if agent is in goal region and will not
                get in the way of previously-planned paths, False 
                else
        """      
        # flag if goal region reached
        found_goal = False

        state = PyBulletTurtle.state_to_euclid(raw_state)
        d = euclidean_distance((state[0], state[1]), (goal[0], goal[1]))
        # if d < goal_radius+agent.radius:
        if d <= goal_radius:
            found_goal = True 
        
        # if an environment was provided, check against moving agents
        # in that env. 
        if found_goal and env is not None:
            for obs in env.obstacles:
                if obs.type is ObsShape.MOVING:
                    # if the obstacle is a moving obstacle, it 
                    # is an agent with a planned path. We must 
                    # check the rest of that path against this 
                    # agent's position to make sure that these 
                    # already-planned agents will not hit it 
                    # as they finish up their paths
                    t_max = obs.max_time
                    t = current_time
                    while t <= t_max:
                        obs_pos = obs.positionAtTime(t)
                        if env.is_collision(agent, raw_state, 
                                            obs.agent, obs_pos):
                            found_goal = False
                            break 
                        t += path_timestep
                # break out of loop over obstacles 
                if not found_goal:
                    break

        return found_goal
    
    # used externally, other agents have to implement this!!!
    @staticmethod
    def state_to_euclid(agent_state):
        """
        Takes a Turtle state tuple and converts it to (x, y)

        Args:
            agent_state (Turtle State tuple): ((x,y,z), (i, j, k, s)) state/quat 
                disaster tuple

        Returns:
            tuple(float,float): basic (x,y) ttuple
        """
        return (agent_state[0][0], agent_state[0][1])
    
    @staticmethod
    def mult_quaternion(qa, qb):
        """
        Multiplies the first PB state attitude quaternion by the second. 

        # MAINT: Order in quaternion multiplication is IMPORTANT
        # MAINT: This may be made more efficient by spelling out the 
            math

        Args:
            qa (tuple(float, float, float, float)): A PB quaternion, (i, j, k, s)
            qb (tuple(float, float, float, float)): A PB quaternion, (i, j, k, s)

        Returns:
            tuple(float, float, float, float): quaternion resulting from multiplying qa by qb,
                (i, j, k, s)
        """
        # numpy quats are (s, i, j, k)
        npa = np.quaternion(qa[3], qa[0], qa[1], qa[2])
        npb = np.quaternion(qb[3], qb[0], qb[1], qb[2])
        res = npa * npb 
        return (res.x, res.y, res.z, res.w)

    @staticmethod
    def rotate_vector(quat, vector):
        """
        Rotates a vector by a PB quaternion

        Args:
            quat (tuple(float, float, float, float)): A PB quaternion, (i, j, k, s)
            vector (tuple(float, float, float)): A PB state tuple (x,y,z)

        Returns:
            tuple(float, float, float): Resulting vector when rotated by 
                the input quat, (x, y, z)
        """
        # numpy quats are (s, i, j, k)
        np_quat = np.quaternion(quat[3], quat[0], quat[1], quat[2])
        np_quat_inv = np.quaternion(quat[3], -quat[0], -quat[1], -quat[2])
        vec_quat = np.quaternion(0., vector[0], vector[1], vector[2])
        res = np_quat * vec_quat * np_quat_inv
        return (res.x, res.y, res.z) 
        
    # used externally, other agents have to implement this!!!
    @staticmethod
    def point_translate_function(base_point, edge_point):
        """
        Translates an edge bundle end state based on a starting state

        Args:
            base_point (agent_state_type): start point for edge propagation 
            edge_end_point (agent_state_type): end of a edge from an edgebundle
                for this agent type

        Returns:
            agent_state_type: edge end point translated based on base_point's 
                position, orientation
        """   
        rot = PyBulletTurtle.rotate_vector(base_point[1], edge_point[0])
        x = base_point[0][0] + rot[0]
        y = base_point[0][1] + rot[1]
        # in a 2d state space, the z component is not really important 
        z = rot[2]

        return ((x, y, z), PyBulletTurtle.mult_quaternion(base_point[1], edge_point[1]))
    
    # used externally, other agents have to implement this!!!
    @staticmethod
    def get_start_from_euclid(twod_start):
        """
        Returns a start state from (x,y), used in test pipeline

        Args:
            twod_start (tuple(float, float)): (x,y) state 

        Returns:
            agent_state_type: twod_start translated to the PB disaster
                state tuple 
        """
        return ((twod_start[0], twod_start[1], 0.1), (0, 0, 0, 1.0))
        

class ObsShape(Enum):
    """
    Classes of PyBullet obstacle 
    """
    RECTANGLE = 1
    CIRCLE = 2
    MOVING = 3

class RectangleObsPybullet(AbstractObstacle):
    def __init__(self, x, y, l, w, h=0.1, color = pybullet_utils.BLACK):
        """
        Get a new PB rectangle obstacle

        Args:
            x (float): obstacle x position
            y (float): obstacle y position
            l (float): obstacle length (x)
            w (float): obstacle width (y)
            h (float, optional): Obstacle height (z). Defaults to 0.1.
            color (list(int), optional): RGBa int list. Defaults to pybullet_utils.BLACK.
        """
        self.x = x 
        self.y = y
        self.l = l
        self.w = w 
        self.h = h
        self.color = color
        self.type = ObsShape.RECTANGLE
        # will instantiate when added to env
        self.pb_id = None 

    # to maintain consistancy, override the default positionAtTime method
    def positionAtTime(self, t):
        return (self.x, self.y, 0.5 * np.sqrt(self.l**2 + self.w**2))
    
class CircleObsPybullet(AbstractObstacle):
    def __init__(self, x, y, r, h=0.1, color = pybullet_utils.BLACK):
        """
        Get a new PB circle obstacle

        Args:
            x (float): obstacle x position
            y (float): obstacle y position
            r (float): obstacle radius
            h (float, optional): Obstacle height (z). Defaults to 0.1.
            color (list(int), optional): RGBa int list. Defaults to pybullet_utils.BLACK.
        """
        self.x = x 
        self.y = y
        self.r = r
        self.h = h
        self.color = color
        self.type = ObsShape.CIRCLE
        # will instantiate when added to env
        self.pb_id = None 

    # to maintain consistancy, override the default positionAtTime method
    def positionAtTime(self, t):
        return (self.x, self.y, self.r)
    

class AgentObsPybullet(AbstractObstacle):
    def __init__(self, agent, rrt):
        """
        An obstacle representation of a pybullet agent moving along
        a path in time

        Args:
            agent (PyBullet agent): Agent to base obstacle from
            rrt: Planner obj for this obstacle 
            states: NOT USED, only here to match the normal 'mathematical' AgentObstacle 
        """
        super().__init__()
        self.state_dict = rrt.get_high_resolution_path()
        self.state_keys = list(self.state_dict.keys())
        self.agent = agent
        self.pb_id = agent.pb_id
        self.type = ObsShape.MOVING
        self.max_time = self.state_keys[-1]
        # these shouldn't be used, but if they are this should be 
        # outside the env
        self.x = -100
        self.y = -100
        self.r = 0

    def positionAtTime(self, t):
        """
        Gets the position of this obstacle at time t

        Args:
            t (float): time to get position of this obs for

        Raises:
            Exception: when t is negative

        Returns:
            PyBullet disaster state tuple: Obstacle position at t
        """
        if (t<0): raise Exception("Time cannot be negative")
        idx = bisect.bisect_left(self.state_keys, t)
        if idx >= len(self.state_keys) - 1: 
            return self.state_dict[self.state_keys[-1]]
        elif t - self.state_keys[idx] < self.state_keys[idx+1] - t:
            return self.state_dict[self.state_keys[idx]]
        else:
            return self.state_dict[self.state_keys[idx+1]]
        




    
