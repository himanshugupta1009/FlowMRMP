from cRRT import *
from numba import njit
from constrainedX import ConstrainedRRT, constraint_satisfaction_numba


def check_collisions_2d_constrained(constrained_crrt, list_of_paths, start_time = None, start_index=0, agent_index=None):
    """
    Collision check func for 'mathematical' sims, i.e. those without high-level 
    sim environments (i.e. PyBullet)

    :MAINT: Assumes circular agents (i.e. agents with a 'radius' field)!

    Args:
        constrained_crrt: constrained crrt obj
        list_of_paths (list(list(agent_state_type))): list of paths for each state, 
            should be equal length and identically spaced
        start_index (int, optional): path index to start checking from (i.e. ignore 
            indices up to this point). Defaults to Zero
        start_time (float, optional): Time to start checking from if not None. If not
            None, used to check against constraints. Defaults to None
        agent_index (int, optional). If set, only check collisions against this agent
            instead of between all agents. Defaults to None

    Returns:
        bool: True if a collision between any two agents detected, false else 
    """

    # if the agent_index is defined, only check against that agent
    first_agent_range = range(len(constrained_crrt.agents))
    if agent_index is not None:
        first_agent_range = range(agent_index, agent_index+1)

    second_agent_range = lambda x : range(x+1, len(constrained_crrt.agents))
    if agent_index is not None:
        second_agent_range = lambda ai : (x for x in range(len(constrained_crrt.agents)) if x != ai)

    if start_time is not None:
        # if start_time is not None, need to check against constraints
        # :MAINT: doing this first to shortcut the process of checking collisions
        #     between agents
        for ind_first_agent in first_agent_range:
            if not constrained_crrt.constraint_satisfaction(list_of_paths[ind_first_agent], start_time):
                return True # there is a collision / constraint violation

    # for each agent...
    for ind_first_agent in first_agent_range:
        first_agent = constrained_crrt.agents[ind_first_agent]
        first_agent_path = list_of_paths[ind_first_agent]
        first_agent_path_len = len(first_agent_path)

        # ...go through all other agents 
        for ind_second_agent in second_agent_range(ind_first_agent):
            second_agent = constrained_crrt.agents[ind_second_agent]
            second_agent_path = list_of_paths[ind_second_agent]
            second_agent_path_len = len(second_agent_path)

            path_len = max(first_agent_path_len, second_agent_path_len)

            # check each state in the paths against each other
            for state_index in range(start_index, path_len):
                state_first_agent = first_agent_path[-1]
                if (state_index < first_agent_path_len):
                    state_first_agent = first_agent_path[state_index]
                state_second_agent = second_agent_path[-1]
                if (state_index < second_agent_path_len):
                    state_second_agent = second_agent_path[state_index]

                # :MAINT: assuming circular agents!!!!
                if point_circle_collision(state_first_agent[0], state_first_agent[1], first_agent.radius, 
                                          state_second_agent[0], state_second_agent[1], second_agent.radius+constrained_crrt.env.obstacle_buffer):
                    # shortcut the process: if one collision is found, all 
                    # paths are invalid 
                    return True # there is a collision
    
    return False # there is no collision

class ConstrainedCRRT(CRRT, ConstrainedRRT):
    def __init__(self, *args, **kwargs):
        start = kwargs.get('starts', [None])[0]
        goal = kwargs.get('goals', [None])[0]
        goal_radius = kwargs.get('goal_radii', [None])[0]
        env = kwargs.get('env', None)
        agent = kwargs.get('agents', [None])[0]
        isvalid_function = kwargs.get('isvalid_function', [None])[0]
        cost_function = kwargs.get('cost_function', [None])[0]
        random_point_function = kwargs.get('random_point_function', [None])[0]
        reached_goal_function = kwargs.get('reached_goal_function', [None])[0]

        ConstrainedRRT.__init__(self, start=start, goal=goal, goal_radius=goal_radius,
                                env=env, agent=agent, isvalid_function=isvalid_function,
                                cost_function=cost_function, random_point_function=random_point_function,
                                reached_goal_function=reached_goal_function)
        CRRT.__init__(self, *args, **kwargs)

        self.prune_tree = False # pruning not supported for constrained cRRTs yet
    
    def extend_tree(self, *args, **kwargs):
        return CRRT.extend_tree(self, *args, **kwargs)
    
    

from cRRT_eb import *

class ConstrainedEdgeBundleType2CRRT(CRRT_EBType2, ConstrainedRRT):
    def __init__(self, *args, **kwargs):
        start = kwargs.get('starts', [None])[0]
        goal = kwargs.get('goals', [None])[0]
        goal_radius = kwargs.get('goal_radii', [None])[0]
        env = kwargs.get('env', None)
        agent = kwargs.get('agents', [None])[0]
        isvalid_function = kwargs.get('isvalid_function', [None])[0]
        cost_function = kwargs.get('cost_function', [None])[0]
        random_point_function = kwargs.get('random_point_function', [None])[0]
        reached_goal_function = kwargs.get('reached_goal_function', [None])[0]

        ConstrainedRRT.__init__(self, start=start, goal=goal, goal_radius=goal_radius,
                                env=env, agent=agent, isvalid_function=isvalid_function,
                                cost_function=cost_function, random_point_function=random_point_function,
                                reached_goal_function=reached_goal_function)
        CRRT_EBType2.__init__(self, *args, **kwargs)

        self.prune_tree = False # pruning not supported for constrained cRRTs yet
    
    def extend_tree(self, *args, **kwargs):
        return CRRT_EBType2.extend_tree(self, *args, **kwargs)

from kinodynamic_TI_eb_crrt import *
class ConstrainedKinoTIEBCRRT(KinoTIEBCRRT, ConstrainedRRT):
    def __init__(self, *args, **kwargs):
        start = kwargs.get('starts', [None])[0]
        goal = kwargs.get('goals', [None])[0]
        goal_radius = kwargs.get('goal_radii', [None])[0]
        env = kwargs.get('env', None)
        agent = kwargs.get('agents', [None])[0]
        isvalid_function = kwargs.get('isvalid_function', [None])[0]
        cost_function = kwargs.get('cost_function', [None])[0]
        random_point_function = kwargs.get('random_point_function', [None])[0]
        reached_goal_function = kwargs.get('reached_goal_function', [None])[0]

        ConstrainedRRT.__init__(self, start=start, goal=goal, goal_radius=goal_radius,
                                env=env, agent=agent, isvalid_function=isvalid_function,
                                cost_function=cost_function, random_point_function=random_point_function,
                                reached_goal_function=reached_goal_function)
        KinoTIEBCRRT.__init__(self, *args, **kwargs)

        self.prune_tree = False # pruning not supported for constrained cRRTs yet
    
    def extend_tree(self, *args, **kwargs):
        return KinoTIEBCRRT.extend_tree(self, *args, **kwargs)