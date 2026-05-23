import sys
sys.path.append('./src')
from Environments import *
from Agents import UniCycle
from rrt import RRT
from sst import SST
from constrainedX import *
from kcbs import *
from mapf_env_square_agent_unicycle import get_unicycle_agent, \
                            get_rrt_planner, get_eb_rrt_planner
from printer import *

starts = [
    (3.163949497924775, 34.646758601910676, 4.9296754203148305),
    (27.89692707137549, 31.038006209376064, 0.6751561527454889),
    (23.19847222488678, 23.75872171697944, 5.758803545472478),
    (26.405248974680568, 22.88349725647492, 1.7213686202918759),
    (14.699737295498325, 17.877880119634472, 0.959323995981249),
    (36.14400315804773, 34.37600356902864, 2.559604396984406),
    (10.665670962196675, 31.0323433962824, 2.1699374865702503),
    (26.636876627482, 8.617495504975288, 3.0220001248398187),
    (34.160966173284024, 7.234946743318524, 2.851217067344404),
    (32.56647692514572, 18.16171971474202, 1.0078897900877453),
]

goals = [
    (15.029748778232639, 25.588067530585423),
    (14.26975060760997, 14.388303212817222),
    (22.36674222556418, 34.23424569856492),
    (10.185386491635853, 29.05230100309256),
    (30.940239842048598, 7.483658086790758),
    (28.423066573468233, 24.39786728738917),
    (22.05701409228253, 25.591090904225386),
    (30.245833206641375, 19.652467316291695),
    (33.7673910994736, 25.119447796108712),
    (17.595487141561495, 18.568599645266556),
]


obstacles = [
    CircularObstacle2D(34.434975589240125, 29.53588919776331, 0.5458094185378046),
    CircularObstacle2D(16.039785337750615, 6.8672839957831044, 3.419598978920261),
    CircularObstacle2D(6.6326523468557985, 8.223918056275759, 2.2391842559861392),
    CircularObstacle2D(9.065285593337345, 16.696079270050365, 1.8388179485506515),
    CircularObstacle2D(6.955614959789056, 23.687155450418214, 1.508728365795669),
    CircularObstacle2D(23.7761740590626, 15.12223955178611, 0.734247814379386),
    CircularObstacle2D(19.560328175754112, 31.2086354673526, 1.0785533950032156),
    CircularObstacle2D(34.10669686726542, 13.50111281792669, 0.5624812884452897),
    CircularObstacle2D(23.646202940023826, 3.8296377826050634, 0.7998783032220491),
    CircularObstacle2D(29.08597529386776, 15.312451743911168, 0.8367239770380241),
    CircularObstacle2D(18.995686804354342, 14.152004403747537, 1.172212085252747),
    CircularObstacle2D(30.089462060902136, 34.964948211602604, 1.5293455793741393),
    CircularObstacle2D(21.181445237069763, 20.218307010459842, 0.5465893764485116),
    CircularObstacle2D(3.5784526236605863, 14.822429195080467, 0.5805351036802748),
    CircularObstacle2D(7.5700838043129135, 33.69158151261289, 1.2435068647221472),
    CircularObstacle2D(24.448498733718203, 29.483284156071228, 1.019717084350062),
    CircularObstacle2D(36.306126303436486, 3.8835971306575696, 0.5645604548750746),
    CircularObstacle2D(18.225708921824847, 24.111851363727606, 0.6765510240718118),
    CircularObstacle2D(22.970203410313438, 11.182599216997184, 0.5134655110980451),
    CircularObstacle2D(28.335789531101174, 3.8913192583428846, 0.651387140231042),
    CircularObstacle2D(35.40831127901773, 21.114276552040785, 0.8195808911952425),
    CircularObstacle2D(14.592044733132191, 33.14999376913684, 1.2067354660592007),
    CircularObstacle2D(3.685106024968513, 28.198872136314286, 0.8471543511366766),
]


env = SquareEnvironment(40.0, 40.0, obstacles)

num_agents = 10
goal_radius = 0.5


agent_ids = []
agents = []
for agent_id in range(num_agents):
    agent_ids.append(agent_id)
    agents.append(get_unicycle_agent(agent_id))

def get_sst_planner(start, goal, goal_radius, agent, env):
    return SST( 
            start=start, goal=goal,
            goal_radius=goal_radius, 
            env = env, agent=agent,
            sampling_time_step=1.5,
            minimum_time_step=0.1,
            max_iter = 10000,
            planning_time=300.0,
            isvalid_function=agent.is_new_node_valid,
            cost_function=agent.get_cost,
            random_point_function=agent.get_random_point,
            reached_goal_function = agent.agent_reached_goal,
            translate_function = agent.kd_tree_point_translate_function,
            sort_edges_function=agent.sort_kd_tree_edges,
            )

sst  = SST( 
            start=start, goal=goal,
            goal_radius=goal_radius, 
            env = env, agent=agent,
            sampling_time_step=1.5,
            minimum_time_step=0.1,
            max_iter = 10000,
            planning_time=300.0,         
            isvalid_function=agent.is_new_node_valid,
            cost_function=agent.get_cost,
            random_point_function=agent.get_random_point, 
            reached_goal_function = agent.agent_reached_goal,
            udf_seed = s,
            print_logs=True,
            # debug_flag=True,
            best_near_radius=5.0
           )

planners = []

planner_function = get_rrt_planner
# planner_function = get_eb_rrt_planner
for i in range(num_agents):
    planners.append(planner_function(starts[i],goals[i],goal_radius,agents[i],env))

