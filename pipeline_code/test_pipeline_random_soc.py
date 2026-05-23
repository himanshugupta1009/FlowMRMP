from contextlib import redirect_stdout
import os

from test_pipeline_random import TestPipeline
from agent_builders import SecondOrderCarBuilder
import numpy as np
from test_classes import *

if __name__ == "__main__":
    # warm up numba
    planning_time = 5.0 
    test_classes = [
        KcbsTestClass(max_planning_time=planning_time), 
        KcbsKinoTiEbTestClass(max_planning_time=planning_time),
        PrrtTestClass(max_planning_time=planning_time),
        PrioritizedKinoTIRRTTestClass(max_planning_time=planning_time),
        CRRTTestClass(max_planning_time=planning_time,
                      branch_goal_parking=True),
        KinoTiCRRTEBTestClass(max_planning_time=planning_time,
                              branch_goal_parking=True,
                              print_logs=False)
        ]
    # collect agent builders
    agent_builders = [SecondOrderCarBuilder(max_speed=1.0, max_acceleration= 2., max_phi=np.pi/3, 
                                            max_steering_rate=0.5, radius=0.3, wheelbase=0.7, kd_num_edges=30000)]
    # instantiate test pipeline 
    tp = TestPipeline(test_classes, agent_builders, test_rounds=1, num_agents=2, master_seed=50)
    tp.run()

    planning_time = 300.0
    for num_agents in [4, 6, 8, 10, 15, 20]:
        for gr in [1.]:
            savepath = "test_results/final_test_results/RANDOM_ALL_SOC_CRRT_UPDATES_A" + str(num_agents) + "_gr" + str(gr)
            os.makedirs(savepath, exist_ok=True) 
            # collect test classes
            test_classes = [
                KcbsTestClass(max_planning_time=planning_time), 
                KcbsKinoTiEbTestClass(max_planning_time=planning_time),
                PrrtTestClass(max_planning_time=planning_time),
                PrioritizedKinoTIRRTTestClass(max_planning_time=planning_time),
                CRRTTestClass(max_planning_time=planning_time,
                              branch_goal_parking=True),
                KinoTiCRRTEBTestClass(max_planning_time=planning_time,
                                      branch_goal_parking=True,
                                      print_logs=False)
                ]
            # instantiate test pipeline 
            tp = TestPipeline(test_classes, agent_builders, test_rounds=100, num_agents=num_agents, master_seed=num_agents*100+5, goal_radius=gr,
                              savepath=savepath)
            with open(savepath+'/log.txt', 'w') as f, redirect_stdout(f):
                tp.run()
            tp.print_stats(filename=savepath, plots=True)
