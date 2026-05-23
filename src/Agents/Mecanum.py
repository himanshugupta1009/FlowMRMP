#Agents.py contains the implementation of the agents that will be used in the simulation.
from collections import namedtuple
import numpy as np
import random
from numba import njit

# from wpimath.kinematics import MecanumDriveKinematics, MecanumDriveWheelSpeeds
# from wpimath.geometry import Translation2d

# from utils import clamp, wrap_between_0_and_2pi, euclidean_distance_numba, \
#     euclidean_distance_satisfaction_numba, euclidean_distance_numba_with_l, \
#     validate_random_point_numba, is_new_node_valid_numba, is_state_valid_numba, \
#     unicycle_point_translate_function_numba, unicycle_point_translate_function_kd_tree_numba, \
#     unicycle_sort_edges_numba, unicycle_sort_kd_tree_edges_numba, \
#     get_distance_covered_numba, check_dynamic_collisions_to_end
    


class Mecanum:
    """
    Uses the wpimath package to simulate the dynamics of a mecanum drive
    agent. 
    """
    def __init__(self, * , agent_id=1, max_speed, radius, rng_seed=11):
        """
        Initializes a new circular Mecanum agent with max wheel speeds and 
        a physical radius 

        agent_id: unique identifier for the agent
        max_speed: maximum speed for each of 4 mecanum wheels
        radius: radius of circular agent
        rng_seed: seed for local random number generator, used to 
                generate random states
        """
        self.id = agent_id
        self.state_length = 3
        self.max_speed = max_speed
        self.radius = radius

        # get x/y values for wheel positions, which will be at the 4 relative
        # "corners" of a circular body
        p = np.sqrt((radius**2)/2)
        fl = Translation2d(p,p)
        fr = Translation2d(p,-p)
        bl = Translation2d(-p,p)
        br = Translation2d(-p,-p)
        # get a kinematics model object, will be used 
        # to model 
        self.kinematics = MecanumDriveKinematics(
            fl, fr, bl, br)

        self.rng = np.random.default_rng(rng_seed)
        self.state_datatype = np.dtype([('x', 'f8'), ('y', 'f8'), ('theta', 'f8')])
        self.state_length = 3

    def move_vehicle(self, state, v, dt):
        """
        Moves the vehicle from a starting state with control 
        inputs v over a period of time dt

        state: start state
        v: 4-tuple control input for each of four wheels
        dt: time delta

        return: new state, 3-tuple of (x, y, heading)
        """
        x, y, theta = state

        # get the state changes from the wheel speed control 
        # input 
        dc = self.kinematics.toChassisSpeeds(
            MecanumDriveWheelSpeeds(v[0], v[1], v[2], v[3])
        )

        new_x = x + dc.vx*dt
        new_y = y + dc.vy*dt
        new_theta = theta + dc.omega*dt

        return (new_x, new_y, new_theta)

    def get_next_state(self, state, control, dt, num_steps=10):
        """
        Move the agent from a start state with a control input for a 
        period of time, returning the final state as well as the path 
        taken to get there over a number of steps. 

        state: start state
        control: 4-tuple control input for each of four wheels
        dt: time delta
        num_steps: number of steps to record in the path 

        return: (new state tuple of (x,y,theta), list of intermediate states size num_steps)
        """

        # path = np.empty(num_steps, dtype=self.state_datatype)
        path = np.empty((num_steps,self.state_length), dtype=np.float64)
        curr_state = state
        exec_dt = dt/num_steps

        for i in range(num_steps):
            new_state = self.move_vehicle(curr_state, control, exec_dt)
            path[i] = new_state
            curr_state = new_state
        
        return curr_state, path
    
    def get_random_action(self,udf_rng):
        """
        Gets a random action within speed bounds

        udf_rng: a random number generator to use for new 
            random control inputs

        return: a 4-tuple of random wheel velocities
        """
        return (udf_rng.uniform(-self.max_speed, self.max_speed),
                udf_rng.uniform(-self.max_speed, self.max_speed),
                udf_rng.uniform(-self.max_speed, self.max_speed),
                udf_rng.uniform(-self.max_speed, self.max_speed))
        
    def get_distance(self, state1, state2):
        """
        Gets a distance between two mecanum agent states

        state1: first agent state
        state2: second agent state

        return: distance between state1 and state2, ignoring orientation 
        """
        # d = euclidean_distance((state1[0], state1[1]), (state2[0], state2[1]))
        return euclidean_distance_numba_with_l(state1, state2, 2)
    
        # return np.linalg.norm(np.array(state1[:2]) - np.array(state2[:2]))

    def check_collision(self, base_agent_state, point):
        """
        Performs a non-rigorous collision check between this
        agent and another agent at a point

        base_agent_state: state of this agent 
        point: state of the other agent 

        return: true if collision detected, false else 
        """
        d = self.get_distance(base_agent_state,point)
        # TODO use other agent radius?
        if d<=self.radius*2.001:
            return True
        else:
            return False
        
    """
    UTILITY FUNCS
    """
    @staticmethod
    def is_state_valid(env, agent, state):
        """
        Check if a single state is valid for an environment 
        
        :PRE: Environment contains circular obstacles ONLY!

        Args:
            env: Environment object 
            agent: Mecanum agent object
            state (Mecanum state tuple): agent location

        Returns:
            _type_: _description_
        """
        # check if agent state x component is within the env
        if(state[0] < (agent.radius + env.boundary_buffer) 
                or state[0] > (env.size[0]-agent.radius-env.boundary_buffer)):
            return False
        # check if agent state y component is within the env
        if(state[1] < (agent.radius + env.boundary_buffer)
                or state[1] > (env.size[1]-agent.radius-env.boundary_buffer)):
            return False
        
        # For each obstacle...
        for obs in env.obstacles:
            # ...check if it is far enough away to not be a collision 
            if obs.check_collision(state, agent.radius, obstacle_buffer=env.obstacle_buffer):
                return False
        return True

    @staticmethod
    def is_new_node_valid(env, agent, path_to_new_state):
        """
        Checks if a new state and the path to that new state is valid for 
        an environment.

        :PRE: Environment contains circular obstacles ONLY!

        Args:
            env: Environment object 
            agent: Mecanum agent object
            path_to_new_state (list[mecanum state tuple]): Path taken by the agent 
                from the parent state, non-inclusive of the parent state but inclusive
                of the final state 

        Returns:
            bool: True if new node is valid, false else 
        """
        # check if each element of the path is valid
        for state in path_to_new_state:
            if not Mecanum.is_state_valid(env, agent, state):
                return False
        return True
    
    @staticmethod
    def is_state_valid_moving_obs(env, agent, state, time):
        """Check if a single state is valid for an environment with moving 
        obstacles (i.e. obstacles with a time dimension)
        
        :PRE: Environment contains circular obstacles with a TIME COMPONENT ONLY!

        Args:
            env: Environment object 
            agent: Mecanum agent object
            state (Mecanum state tuple): agent location
            time (float): current time, fractional seconds 

        Returns:
            bool: True if new state valid, false else 
        """
        # check if agent state x component is within the env
        if(state[0] < (agent.radius + env.boundary_buffer) 
                or state[0] > (env.size[0]-agent.radius-env.boundary_buffer)):
            return False
        # check if agent state y component is within the env
        if(state[1] < (agent.radius + env.boundary_buffer)
                or state[1] > (env.size[1]-agent.radius-env.boundary_buffer)):
            return False
        
        # For each obstacle...
        for obs in env.obstacles:
            # check if it is far enough away to not be a collision 
            if obs.check_collision(state, agent.radius, t=time, obstacle_buffer=env.obstacle_buffer):
                return False
        return True

    @staticmethod
    def is_new_node_valid_moving_obs(env, agent, path_to_new_state, start_time, interval, 
                                     check_goal = False):
        """Checks if a new state and the path to that new state is valid for 
        an environment with moving obstacles (i.e. obstacles with a time dimension)

        :PRE: Environment contains circular obstacles with a TIME COMPONENT ONLY!

        Args:
            env: Environment object 
            agent: Mecanum agent object
            path_to_new_state (list[Mecanum state tuple]): Path taken by the agent 
                from the parent state, non-inclusive of the parent state but inclusive
                of the final state 
            start_time (float): timepoint agent leaves the parent state
            interval (float): length of time the agent takes to complete the path 
            check_goal (bool, optional) Check if this state is valid as a final state
                for the agent. Defaults to False

        Returns:
            bool: True if new node is valid, false else 
        """
        # find the interval of time between each step in the path
        dt = interval/len(path_to_new_state)
        # since the path is not inclusive of the parent node state, 
        # the time for the first item in the path is one step away 
        # from the start time 
        t = start_time + dt
        # for each state...
        for state in path_to_new_state:
            # check if the path entry is not colliding with the 
            # env boundary or any obstacles 
            if not Mecanum.is_state_valid_moving_obs(env, agent, state, t):
                return False
            t += dt

        is_new_state_valid = True
        if check_goal:
            # check against moving obs
            # in the env.
            print("GOAL CHECK MECANUM") 
            state = path_to_new_state[-1]
            current_time = start_time+interval 
            for obs in env.obstacles:
                # if the obstacle is a moving obstacle, i.e.
                # the max_time is greater than 0, it 
                # is an agent with a planned path. We must 
                # check the rest of that path against this 
                # agent's position to make sure that these 
                # already-planned agents will not hit it 
                # as they finish up their paths
                t_max = obs.max_time
                t = current_time
                while t <= t_max:
                    if obs.check_collision(state, agent.radius, t=t, obstacle_buffer=env.obstacle_buffer):
                        is_new_state_valid = False
                        break 
                    t += dt
                # break out of loop over obstacles 
                if not is_new_state_valid:
                    break

        return is_new_state_valid
    
    @staticmethod
    def get_cost(env, agent, parent_state, a, t, path): 
        """
        Gets the cost a Mecanum agent incurs to traverse 
        a path

        :MAINT: Arg list with unused elements to maintain 
        consistency with other agents 

        Args:
            env: Environment object 
            agent: Mecanum agent object
            parent_state (mecanum_agent_state_type): Start point for path 
                NOT USED
            a (mecanum_agent_control_type): control input that generated
                path NOT USED
            t (float): Time over which path was propagated NOT USED
            edge (list(mecanum_agent_state_type)): new path over which 
                to generate cost

        Returns:
            float: approximate path cost
        """        
        path_dist_approx = 0.
        # start state is not included in path 
        last_state = parent_state
        for path_state in path:
            # path_dist_approx += euclidean_distance((path_state[0], path_state[1]), 
            #                                       (last_state[0], last_state[1]))
            path_dist_approx = euclidean_distance_numba_with_l(path_state, last_state, 2)
            last_state = path_state
        return path_dist_approx
            
    @staticmethod
    def get_random_point(env, agent, rng):
        """
        Gets a random point for a mecanum agent in an environment 


        Args:
            env: Environment object 
            agent: Mecanum agent object
            rng (numpy.random): random number generator

        Returns:
            tuple(x, y): new valid point for the agent in the environment  
        """        
        x = rng.uniform(0, env.size[0])
        y = rng.uniform(0, env.size[1])
        p = (x, y)

        while not Mecanum.is_state_valid(env, agent, p):
            x = rng.uniform(0, env.size[0])
            y = rng.uniform(0, env.size[1])
            # theta = rng.uniform(0, 2 * np.pi)
            # p = (x, y, theta)
            p = (x, y)
            
        return p 
    
    @staticmethod
    def agent_reached_goal(state, goal, goal_radius, agent):
        """
        Determines whether the agent has reached the goal region

        :MAINT: Arg list with unused elements to maintain 
        consistency with other agents 

        Args:
            state (agent_state_type): agent state
            goal (tuple(float, float)): goal center
            goal_radius (float): goal region radius
            agent: Mecanum agent object NOT USED

        Returns:
            _type_: _description_
        """        
        return euclidean_distance_satisfaction_numba(state, goal, goal_radius)        
    
    @staticmethod
    def point_translate_function(base_point, edge_end_point):
        """
        Translates an edge bundle end state based on a starting state

        Args:
            base_point (mecanum_agent_state_type): start point for edge propagation 
            edge_end_point (mecanum_agent_state_type): end of a edge from an edgebundle
                for this agent type

        Returns:
            mecanum_agent_state_type: edge end point translated based on base_point's 
                position, orientation
        """        
        #rotation_matrix = [cos(theta) -sin(theta); sin(theta) cos(theta)]
        theta = base_point[2]
        new_edge_end_point_x = edge_end_point[0]*np.cos(theta) - edge_end_point[1]*np.sin(theta)
        new_edge_end_point_y = edge_end_point[0]*np.sin(theta) + edge_end_point[1]*np.cos(theta)
        return (base_point[0] + new_edge_end_point_x, base_point[1] + new_edge_end_point_y, theta + edge_end_point[2])

"""
# Example instantiation, usage of the Mecanum Agent

import sys
sys.path.append('./src')
from Agents import *

# instantiate new agent
m = Mecanum(agent_id=1, 
            max_speed=5, 
            radius=1,
            rng_seed=42)

# move from a start state to new state
# 's' using a control input
state = (0,0,0)
control = m.get_random_action(m.rng)
dt = 0.5
s, path = m.get_next_state(state, control, dt)
print(s)
"""   
