import sys
sys.path.append('.')
sys.path.append('src/')

from pybullet_env import PyBulletTurtle, PyBulletEnv
import importlib  
pybullet_utils = importlib.import_module("pybullet-planning.pybullet_tools.utils")

import argparse
from datetime import datetime
import random
import numpy as np
from tqdm import tqdm
import os

"""
Generates an edge bundle package for the Turtlebot agent in 
a PyBullet environment 
"""

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--num-samples', type=int, default=10000)
    parser.add_argument('--num-batches', type=int, default=1)
    args = parser.parse_args()
    plot_graphs = False
    rng = np.random.default_rng(101)
             
    # set up agent/env
    agent_speed = 10
    agent = PyBulletTurtle(10, 1)
    env = PyBulletEnv(10, 10, [], speed=240, draw_borders=False, use_gui=False)
    env.add_agent(agent, start = [0, 0, 0.0132])

    del_t = 1.0 / 10.  #control frequency
    for _ in range(args.num_batches):       
        progress_bar = tqdm(total=args.num_samples)
        fin_states = np.empty(args.num_samples, dtype=agent.state_datatype)
        fin_trajectories = np.empty(args.num_samples, dtype=object)
        actions = []
        times = []

        propogations = []
        no_of_successfully_generated_edges = 0
        while no_of_successfully_generated_edges < args.num_samples:
            new_start_state = ((0, 0, 0.0132), (0, 0, 0, 1.0))
            # random_linear_velocity = np.random.uniform(-2,2)
            list_of_times = [1,1.1,1.2,1.3,1.4,1.5,1.6,1.7,1.8,1.9,2]
            random_delta_t = random.choice(list_of_times)
            # random_angular_velocity = np.random.uniform(-math.pi/2,math.pi/2)
            #no_of_steps = 10
            minimum_time_step=0.1
            num_record_steps = round(random_delta_t/minimum_time_step)
            random_action = agent.get_random_action(rng)
            new_state, trajectory = agent.get_next_state(new_start_state, random_action, 
                                                         random_delta_t, num_record_steps)

            fin_states[no_of_successfully_generated_edges] = trajectory[-1]
            fin_trajectories[no_of_successfully_generated_edges] = trajectory
            actions.append(random_action)
            times.append(random_delta_t)
            no_of_successfully_generated_edges+= 1
            progress_bar.update(1)
            # end loop over batch
        progress_bar.close()
        np.savez_compressed('eb_pb_turtle_speed_'+str(agent_speed)+'_edges-%d.npz' % (args.num_samples),
                    final_states=fin_states,
                    actions=actions,
                    timesteps=times)
        print('Saved data')
        # end batch loop
        
    