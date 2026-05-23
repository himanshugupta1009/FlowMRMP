import time 
from numba.typed import List
from numba import types
from rrt import RRT
import numpy as np


class PrioritizedPlanning:
    
    @staticmethod
    def plan_multi(*, planners: list[RRT], 
                    planning_time=300.0, 
                    print_logs=False):
        """Perform Priority Planning by a list of ordered planners
        Each agent plans its path individually in a priority order. Subsequent agents 
        must plan using the previous agents' positions along their paths as obstacles. 

        Args:
            planners (list<RRT>): list of planners conforming to the RRT interface
            planning_time (float, optional): maximum time to plan before failure. Defaults to 300.0.
            print_logs (bool, optional): Print information from RRT process. Defaults to False.
        """
        #List of dynamic obstacles that will be updated with each planned path
        dyn_obs = List.empty_list(types.Array(types.float64, 2, 'C'))

        # total costs, will be updated with each planned path
        total_costs = 0.0

        start_time = time.time()
        # iterate over each agent, planning for it
        for i, planner in enumerate(planners):
            if(print_logs):
                print("Planning for agent", i, "with id", planner.agent.id)

            # set the dynamic obstacles to include previously-planned
            # agent paths
            planner.dynamic_agent_obstacles = dyn_obs
            # set the planning time to the remaining total planning time
            planner.planning_time = planning_time - (time.time() - start_time)

            #Plan the path!
            planner.plan_path()

            #Add new agent obstacle if there was a path found for this agent 
            if(planner.path_found and time.time() - start_time <= planning_time):
                total_costs += planner.path_cost

                hires_path = planner.get_high_resolution_path_numpy_array()
                radius_index = planner.agent.distance_metric_state_size
                # hack to include radius in path
                if hires_path.shape[1] <= radius_index:
                    dyn_path = np.empty((hires_path.shape[0], radius_index + 1), dtype=np.float64)
                    dyn_path[:, :radius_index] = hires_path[:, :radius_index]
                    dyn_path[:, radius_index] = planner.agent.radius
                    hires_path = dyn_path
                else:
                    hires_path[:, radius_index] = planner.agent.radius
                dyn_obs.append(hires_path)
            else:
                end_time = time.time() - start_time 
                print("Prioritized Planning low-level planner failed to find a path for agent", i, 
                      "Failed to solve the current MRMP problem using pRRT.")
                return False, end_time, np.inf

        end_time = time.time() - start_time

        return True, end_time, total_costs


# Backwards-compatible alias for older imports.
PrioritizedPlaning = PrioritizedPlanning
