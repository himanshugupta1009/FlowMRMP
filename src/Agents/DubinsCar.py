#Agents.py contains the implementation of the agents that will be used in the simulation.
from collections import namedtuple
import numpy as np
import random
from numba import njit

from utils import clamp, wrap_between_0_and_2pi, euclidean_distance_numba, \
    euclidean_distance_satisfaction_numba, euclidean_distance_numba_with_l, \
    validate_random_point_numba, is_new_node_valid_numba, is_state_valid_numba, \
    get_distance_covered_numba, check_dynamic_collisions_to_end
    

# DubinsCarState = namedtuple('DubinsCarState', ['x','y','theta','v'])
class DubinsCar:
    def __init__(self, * , agent_id=1, wheelbase, max_speed, max_steering_angle, radius, rng_seed=11):
        self.agent_id = agent_id
        self.state_length = 4
        self.L = wheelbase
        self.max_speed = max_speed
        self.max_steering_angle = max_steering_angle
        self.radius = radius
        self.rng = np.random.default_rng(rng_seed)
        self.state_datatype = np.dtype([('x', 'f8'), ('y', 'f8'), ('theta', 'f8'), ('v', 'f8')])
        self.state_length = 4
        self.distance_metric_state_size = 2


    def equation_of_motion(self, state, control):
        x, y, theta, v = state
        acceleration, steering_angle = control
        x_dot = v * np.cos(theta)
        y_dot = v * np.sin(theta)
        theta_dot = v * np.tan(steering_angle)
        v_dot = acceleration
        return (x_dot, y_dot, theta_dot, v_dot)

    def move_vehicle(self, state, new_v, steering_angle , dt):

        x, y, theta, v = state

        if(steering_angle == 0.0):
            new_theta = theta + (new_v*np.tan(steering_angle)*dt/self.L) 
            new_x = x + new_v*np.cos(theta)*dt
            new_y = y + new_v*np.sin(theta)*dt
        else:
            new_theta = theta + (new_v*np.tan(steering_angle)*dt/self.L)
            new_theta = wrap_between_0_and_2pi(new_theta)
            new_x = x + (self.L/np.tan(steering_angle))*(np.sin(new_theta) - np.sin(theta))
            new_y = y + (self.L/np.tan(steering_angle))*(np.cos(theta) - np.cos(new_theta))

        return (new_x, new_y, new_theta, new_v)

    def get_next_state(self, state, control, dt, num_steps=10):

        x, y, theta, v = state
        delta_v, steering_angle = control
        new_v = clamp(v + delta_v, 0.0, self.max_speed)

        # path = np.empty(num_steps, dtype=self.state_datatype)
        path = np.empty((num_steps,self.state_length), dtype=np.float64)
        curr_state = state
        exec_dt = dt/num_steps

        for i in range(num_steps):
            new_state = self.move_vehicle(curr_state, new_v, steering_angle, exec_dt)
            path[i] = new_state
            curr_state = new_state
        
        return curr_state, path
    
    def get_random_action(self,udf_rng):
        delta_v = udf_rng.uniform(-self.max_speed, self.max_speed)
        steering_angle = udf_rng.uniform(-self.max_steering_angle, self.max_steering_angle)
        return (delta_v, steering_angle)
        # return (delta_v, steering_angle)
        
    def get_distance(self, state1, state2):
        # return euclidean_distance((state1[0], state1[1]), (state2[0], state2[1]))
        return euclidean_distance_numba_with_l(state1, state2, self.distance_metric_state_size)


"""
# Example instantiation, usage of the DubinsCar Agent

import sys
sys.path.append('./src')
from Agents import *

d = DubinsCar(agent_id=1,
            wheelbase=1.0,
            max_speed=2.0,
            max_steering_angle=np.pi/4,
            radius=1.0,
            rng_seed=77
            )
state = (0,0,0,0)
control = (1,np.pi/6)
dt = 0.5
state, path = d.get_next_state(state, control, dt)

"""            


