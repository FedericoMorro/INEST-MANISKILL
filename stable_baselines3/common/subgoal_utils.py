def subgoals_list_to_perc_dict(subgoals_list, max_subgoal, n_episodes):
    subgoals_dict = {i: 0 for i in range(max_subgoal + 1)}
    for subgoal in subgoals_list:
        # non-cumulative for 0
        if subgoal == 0:
            subgoals_dict[0] += 1
        # cumulative for others
        else:
            for i in range(1, subgoal + 1):
                subgoals_dict[i] += 1

    # normalize by number of episodes
    for subgoal in subgoals_dict:
        subgoals_dict[subgoal] /= n_episodes
        
    return subgoals_dict