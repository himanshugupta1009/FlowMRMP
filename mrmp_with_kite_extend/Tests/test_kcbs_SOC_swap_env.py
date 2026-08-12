import sys

sys.path.insert(0, "./src")
sys.path.insert(0, "./pipeline_code")

from agent_builders import SecondOrderCarBuilder
from test_classes import KcbsKinoTiEbTestClass, KcbsTestClass
from test_pipeline_swap import TestPipelineOpen


num_agents = 10
goal_radius = 0.5
env_width = 20
env_depth = 20
max_planning_time = 300.0
# True: only accept a terminal goal state when remaining there is safe; retain
# an unsafe goal arrival as a transit state so planning can continue from it.
# False: use the earlier planner-specific behavior. Constrained RRT keeps its
# existing future dynamic-obstacle check and discards a blocked endpoint;
# constrained KiTE accepts a valid arrival without a stationary-tail check.
use_goal_parking_fix = True

seeds = [7]
# seeds = [2000 + i for i in range(100)]

# Select the low-level planner by leaving one assignment active. These test
# classes contain the same planner parameters used by the SWAP pipeline.
test_class = KcbsTestClass(
    max_planning_time=max_planning_time,
    obs_buffers=False,
    use_goal_parking_fix=use_goal_parking_fix,
)
# test_class = KcbsKinoTiEbTestClass(
#     max_planning_time=max_planning_time,
#     obs_buffers=False,
#     use_goal_parking_fix=use_goal_parking_fix,
# )

pipeline = TestPipelineOpen(
    test_classes=[test_class],
    agent_builders=[SecondOrderCarBuilder()],
    test_rounds=len(seeds),
    num_agents=num_agents,
    master_seed=seeds[0],
    env_width=env_width,
    env_bredth=env_depth,
    goal_radius=goal_radius,
    processes=1,
)

costs = []
time_array = []
successes = []

print("Starting KCBS Planning Experiments")
print("Planner:", test_class.name)
print("Number of agents:", num_agents)
print("#######################################################")

for seed_index, seed in enumerate(seeds):
    agents, starts, obstacles, goals, goal_radii = pipeline.get_env_parms(seed)
    success, planning_time, cost, average_path_time, max_path_time, message = (
        test_class.test_func(
            agents,
            starts,
            obstacles,
            goals,
            goal_radii,
            seed,
            env_width,
            env_depth,
        )
    )

    print("Iteration:", seed_index)
    print("RNG Seed:", seed)
    print("Success:", success)
    print("Time taken for planning:", planning_time)
    print("Cost:", cost)
    print("Average agent path time:", average_path_time)
    print("Maximum agent path time:", max_path_time)
    print(message)
    print("#######################################################")

    successes.append(success)
    time_array.append(planning_time)
    costs.append(cost)
