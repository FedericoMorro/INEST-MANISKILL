def subgoals_list_to_perc_dict(subgoals_list, max_subgoal, n_episodes, is_list_of_idxs=False):
    subgoals_dict = {i: 0 for i in range(max_subgoal + 1)}
    
    for subgoal in subgoals_list:
        # if it's a list of subgoal idxs, we take the length as the current subgoal, otherwise we take the value directly
        curr_subgoal = len(subgoal) if is_list_of_idxs else subgoal
        # non-cumulative for 0
        if curr_subgoal == 0:
            subgoals_dict[0] += 1
        # cumulative for others
        else:
            for i in range(1, curr_subgoal + 1):
                subgoals_dict[i] += 1

    # normalize by number of episodes
    for subgoal in subgoals_dict:
        subgoals_dict[subgoal] /= n_episodes
        
    return subgoals_dict