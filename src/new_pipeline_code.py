agents = {}
for agent_id in range(len(starts)):
    agents[agent_id] = get_unicycle_agent(agent_id)

planner_function = get_rrt_planner
planner_function = get_eb_rrt_planner

cost_arr = []
time_arr = []
num_experiments = 50

for k in range(num_experiments):

    planners = {}
    for i in range(len(starts)):
        planners[agents[i].id] = planner_function(starts[i],goals[i],goal_radii[i],agents[i],env)

    kcbs_planner = KCBS(
                    env = env,
                    agents = agents,
                    low_level_planners = planners,
                    max_trials = 1000,
                    planning_time = 100.0
                    )

    st = time.time()
    path_found, paths, path_cost = kcbs_planner.plan_multi_agent_paths()
    td = time.time() - st
    print("************************************************************************")
    print("Cost for iteration ", k+1, " is : ",path_cost)
    cost_arr.append(path_cost)
    print("Time for iteration ", k+1, " is : ",td)
    print("************************************************************************")
    time_arr.append(td)
                    
mc = np.mean(cost_arr)
sdc = np.std(cost_arr)
print( "Mean Cost: ", mc, " SDE: ", sdc/np.sqrt(num_experiments) )


mt = np.mean(time_arr)
sdt = np.std(time_arr)
print( "Mean Time: ", mt, " SDE: ", sdt/np.sqrt(num_experiments) )


#Cost using RRT KCBS
# Mean Cost:  311.8246239025071  SDE:  3.007948073813698
# Mean Time:  48.08456768989563  SDE:  10.781107208225016

#Cost using EBRRT KCBS
# Mean Cost:  273.8622240496205  SDE:  3.715332698570755
# Mean Time:  24.889118432998657  SDE:  6.2084291472703