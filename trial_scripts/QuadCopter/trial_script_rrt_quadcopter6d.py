import sys
sys.path.append('./src')
import time
import numpy as np

from Environments import CuboidEnvironment, SphericalObstacle3D
from Agents import QuadCopter6D
from rrt import RRT
from mapf_env_cuboid_agent_quadcopter6d import get_quadcopter_agent
from visualizations.quadcopter_visualization import visualize_quadcopter_path


obstacles = [
            SphericalObstacle3D(x=2.0, y=2.0, z=0.8, r=0.4),
            SphericalObstacle3D(x=4.0, y=3.0, z=1.4, r=0.5),
            ]
# obstacles = []
env = CuboidEnvironment(length=14.0, breadth=14.0, height=6.0, obs=obstacles)
start = np.array((13.0, 1.0, 0.6, 0.0, 0.0, 0.0), dtype=np.float64)
goal = np.array((1.0, 13.0, 4.9), dtype=np.float64)
goal_radius = 0.3


agent = get_quadcopter_agent(agent_id=1)


s = np.random.randint(0, 1000)
rrt = RRT(
            start=start,
            goal=goal,
            goal_radius=goal_radius,
            env=env,
            agent=agent,
            sampling_time_step=1.0,
            use_fixed_sampling_time=False,
            minimum_time_step=0.1,
            max_iter=10000,
            num_extension_trials=10,
            planning_time=300.0,
            isvalid_function=agent.is_new_node_valid,
            cost_function=agent.get_cost,
            random_point_function=agent.get_random_point,
            reached_goal_function=agent.agent_reached_goal,
            udf_seed=s,
            print_logs=True,
            debug_flag=False
        )

start_time = time.time()
rrt.plan_path()
end_time = time.time()
t = end_time - start_time
print(f"Time taken = {t:.2f} seconds and seed = {s}")
rrt_node_ids, states, actions, timesteps = rrt.get_path()


# Toggle visualization for this test file
VISUALIZE_QUAD = False
if VISUALIZE_QUAD and rrt.path_found:
    path_dict = rrt.get_high_resolution_path()
    visualize_quadcopter_path(path_dict, env, agent_radius=agent.radius)



"""


from quadcopter_viz_3d import (
    save_quadcopter_rrt_tree_3d,
    save_quadcopter_path_3d,
    save_quadcopter_path_overlay_3d,
)

save_quadcopter_rrt_tree_3d(planner, "media/my_tree_3d.png", print_tree=True)

raw_states, _, _ = planner.get_high_resolution_path_and_actions()
save_quadcopter_path_3d(
    env=planner.env,
    start=planner.start,
    goal=planner.goal,
    goal_radius=planner.goal_radius,
    highres_states=raw_states,
    filename="media/my_path_3d.png",
    agent_radius=planner.agent.radius,
    path_color="xkcd:orange",
    path_label="Raw path",
)

save_quadcopter_path_overlay_3d(
    planner,
    opt_result.path_view,
    "media/my_overlay_3d.png",
)


"""