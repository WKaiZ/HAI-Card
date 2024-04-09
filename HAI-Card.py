import pandas as pd
import random
import numpy as np
from linear_irl import large_irl
from value_iteration import value

def get_states(cards, height, memo = {}):
    if cards == [0] * height:
        return {f'0 ' * height}
    returnedCards = set()
    if tuple(cards) in memo:
        return memo[tuple(cards)]    
    for i, card in enumerate(cards):
        if card != 0:
            nextCards = cards.copy()
            nextCards[i] //= 10
            returnedCards.add(' '.join([str(i) for i in nextCards]))
            returnedCards |= get_states(nextCards, height, memo)
    memo[tuple(cards)] = set(returnedCards)
    return set(returnedCards)

def get_next_state(state_str, action):
    state = state_str.split()
    state = [int(i) for i in state]
    row_to_mod = state[action - 1]
    if row_to_mod == 0:
        return None
    row_to_mod //= 10
    state[action - 1] = row_to_mod
    return ' '.join([str(i) for i in state])

def get_transition_probabilities(states, num_actions):
    returned_lst_1 = []
    for i in states:
        returned_lst_2 = []
        next_states = [get_next_state(i, action + 1) for action in range(num_actions)]
        num_of_nones = sum([state == None for state in next_states])
        for j in range(num_actions):
            returned_lst_3 = []
            if next_states[j] == None:
                returned_lst_2.append([0] * len(states))
            else:
                for k in states:
                    if k == next_states[j]:
                        returned_lst_3.append((1 / (num_actions - num_of_nones - 1 + 8)) * 8)
                    elif k in next_states:
                        returned_lst_3.append(1 / (num_actions - num_of_nones - 1 + 8))
                    else:
                        returned_lst_3.append(0)
                returned_lst_2.append(returned_lst_3)
        returned_lst_1.append(returned_lst_2)
    return np.array(returned_lst_1)

def get_policy(states, num_actions, initial_state, elimination_sequence):
    state = " ".join([str(i) for i in initial_state])
    policy_list = [random.randint(0, num_actions - 1) for _ in range(len(states))]
    elimination_sequence = [int(i) for i in elimination_sequence]
    for i in elimination_sequence:
        index = list(states).index(state)
        policy_list[index] = i
        state = get_next_state(state, i + 1)
    return policy_list    

def get_values(feature_matrix, policy, num_states, transition_probabilities, discount_factor = 0.9, threshold = 0.1):
    values = []
    for dim in range(feature_matrix.shape[1]):
        reward = feature_matrix[:, dim]
        values.append(value(policy, num_states, transition_probabilities, reward, discount_factor, threshold))
    return np.array(values)

def get_feature_matrix(states):
    lst = []
    for i in range(len(states)):
        feature = np.zeros(len(states))
        feature[i] = 1
        lst.append(feature)
    return np.array(lst)

if __name__ == '__main__':
    states = get_states([131242, 412323, 142243, 443131], 4)
    states = states.union({'131242 412323 142243 443131'})
    print(len(states))
    print('Done getting states')
    num_states = len(states)
    actions = 4
    transition = get_transition_probabilities(states, actions)
    print('Done getting transition probabilities')
    policy = get_policy(states, actions, [131242, 412323, 142243, 443131], '211100030111222303322033')
    print('Done getting policy')
    feature_matrix = get_feature_matrix(states)
    print('Done getting feature matrix')
    values = get_values(feature_matrix, policy, num_states, transition)
    print('Done getting values')
    rewards = large_irl(values, transition, feature_matrix, num_states, actions, policy)
    print('Done getting rewards')
    print(rewards)
