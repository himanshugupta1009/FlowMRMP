import sys
sys.path.append("./src")
from Environments import SquareEnvironment, CircularObstacle2D
from mapf_env_square_agent_unicycle import get_unicycle_agent
from sst import SST
from sst_printer import SSTPrinter
import numpy as np


obstacles = [
            CircularObstacle2D(10, 10, 2),
            CircularObstacle2D(16, 25, 3),
            CircularObstacle2D(20, 5, 2),
            CircularObstacle2D(35, 15, 4),
            CircularObstacle2D(30, 34, 4),
            CircularObstacle2D(25, 15, 4),
            CircularObstacle2D(7, 19, 5),
            CircularObstacle2D(16, 16, 2),
            CircularObstacle2D(33, 4, 2),
            CircularObstacle2D(8, 34, 3),
            CircularObstacle2D(20, 32, 2),
            CircularObstacle2D(31, 24, 3),
            ]
obstacles= []
env = SquareEnvironment(40, 40, obstacles, obs_buffers=False)
agent = get_unicycle_agent(agent_id = 1)
start = np.array([7.0, 5.0, 0.0])
goal = np.array([24.0, 37.0])
# goal = (25.0, 25.0)
goal_radius = 0.5


s = np.random.randint(0, 1000)
sst  = SST( 
            start=start, goal=goal,
            goal_radius=goal_radius, 
            env = env, agent=agent,
            sampling_time_step=2.0,
            use_fixed_sampling_time=False,
            minimum_time_step=0.1,
            max_iter = 10000,
            planning_time=300.0,
            num_extension_trials=10,
            goal_sampling_probability=0.1,
            isvalid_function=agent.is_new_node_valid,
            cost_function=agent.get_cost,
            random_point_function=agent.get_random_point, 
            reached_goal_function = agent.agent_reached_goal,
            udf_seed = s,
            print_logs=True,
            # debug_flag=True,
            best_near_radius=2.0
           )

sst.plan_path()
print("First solution planning time:", sst.first_solution_planning_time)
print("First solution path cost:", sst.first_solution_path_cost)
print("Final path cost:", sst.path_cost)
sst.path_time
node_ids, states, actions, timesteps = sst.get_path()

printer = SSTPrinter(env, sst)
printer.print_sst("media/sst_unicycle_graph.png",
                  show_tree=True, show_path=True,
                  show_active=True, show_inactive=False,
                  show_witness=True, show_prune_circles=False)


# optional animation
# printer.print_sst_step_ani("media/sst_build.gif", animation_speed=2)

