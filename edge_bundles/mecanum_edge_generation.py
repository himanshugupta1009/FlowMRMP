
import argparse
from copy import copy
from datetime import datetime
import math
from sys import path
import random
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

# import the Mecanum agent
import sys
sys.path.append('.')
sys.path.append('src/')
from Agents import Mecanum

"""
Generates an edge bundle package for the Mecanum agent
"""

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--num-samples', type=int, default=100000)
    parser.add_argument('--num-batches', type=int, default=1)
    args = parser.parse_args()
    plot_graphs = False
    if plot_graphs:
            fig, ax = plt.subplots(1,1)
            ax.set_xlim([-20, 20])
            ax.set_ylim([-20, 20])
             
    # set up agent/env
    agent_speed = 5
    agent_radius = 1
    agent = Mecanum(max_speed=agent_speed, radius=agent_radius)
 
    # random times to generate edges over 
    list_of_times = [1,1.1,1.2,1.3,1.4,1.5,1.6,1.7,1.8,1.9,2]
    # minimum time step over path trajectory
    minimum_time_step=0.1

    del_t = 1 / 10  #control frequency
    for _ in range(args.num_batches):
        progress_bar = tqdm(total=args.num_samples)
        # edge info 
        fin_states = []
        actions = []
        times = []
        
        
        propogations = []
        no_of_successfully_generated_edges = 0
        while no_of_successfully_generated_edges < args.num_samples:
            new_start_state = np.array([0.,0.,0.])
            
            # get a new random time from the set of allowable choices
            random_delta_t = random.choice(list_of_times)
            # get resulting state and path to that state
            num_record_steps = round(random_delta_t/minimum_time_step)
            control = agent.get_random_action(agent.rng)
            s, trajectory = agent.get_next_state(new_start_state, control, random_delta_t)

            # save off resulting state
            new_start_state = np.array([s[0], s[1], s[2]])

            # plot if necessary 
            if plot_graphs:
                blue = math.sqrt(control[0]**2 + control[1]**2 + control[2]**2 + control[3]**2)/(6)
                if blue > 1:
                    blue = 1
                new_x = [trajectory[i][0] for i in range(len(trajectory))]
                new_y = [trajectory[i][1] for i in range(len(trajectory))]
                ax.plot(new_x,new_y,'o', markersize = 5, color = (0,0,blue))
                plt.pause(0.05)

            # save off this loop's states to add to final bundle 
            fin_states.append([trajectory[-1][0], trajectory[-1][1], trajectory[-1][2]])
            actions.append(control)
            times.append(random_delta_t)
            no_of_successfully_generated_edges+= 1
            progress_bar.update(1)
        progress_bar.close()
        file_timestamp = datetime.now().strftime('%Y%m%d-%H%M%S')
        np.savez_compressed('eb_mecanum_r'+str(agent_radius)+'_s'+str(agent_speed)+'_edges-%d_%s.npz' % (args.num_samples, file_timestamp),
                    final_states= fin_states,
                    actions=actions,
                    timesteps = times)

        print('Saved data')
        plt.show()