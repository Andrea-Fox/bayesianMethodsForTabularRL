from operator import mul
import gym
import numpy as np # 1. Load Environment and Q-table structure
import math 
import pandas as pd
import matplotlib.pyplot as plt

from gym import spaces
from gym.utils import seeding

import multiprocessing as mp

from mountain_car_env import MountainCarEnv

def argmaxrand(a):
    '''
    If a has only one max it is equivalent to argmax, otehrwise it uniformly random selects a maximum
    '''
    indeces = np.where(np.array(a) == np.max(a))[0]
    return np.random.choice(indeces)

def boltzmann_exploration(q_values, state, state_counts = 0, exploration_rate = 0):
    # we choose C_t equal to the maximum between the values
    C = exploration_rate * max(abs(q_values[state, 0] - q_values[state, 1]), abs(q_values[state, 0] - q_values[state, 1]), abs(q_values[state, 1] - q_values[state, 2]) )
    if C==0:
        C = 1
    state_counts[state] += 1

    exponentials = [math.exp( (math.log(state_counts[state])/C) *  q_values[state, action]   )   for action in range(3)]
    sum_exponentials = sum(exponentials)
    if sum_exponentials == 0:
        sum_exponentials = 1
        exponentials = [1/3, 1/3, 1/3]
    probabilites_vector = [exponentials[action]/sum_exponentials  for action in range(3) ]

    # now we need to draw an action according to the probabilty given by probabilites_vector
    action = np.random.choice(np.arange(0, 3), p = probabilites_vector)
    return action, state_counts

def egreedy_policy(q_values, state, epsilon=0.1, state_counts = 0):
    ''' 
    Choose an action based on a epsilon greedy policy.    
    A random action is selected with epsilon probability, else select the best action.    
    '''
    state_counts[state] += 1
    if np.random.random() < epsilon:
        return np.random.choice(3), state_counts
    else:
        # print(np.argmax(q_values[state, ]))
        return argmaxrand(q_values[state, :]), state_counts

def q_learning(env, num_episodes=500, render=True, exploration_rate=0.1, learning_rate=0.5, discount_factor=0.9, learning_policy = "epsilon-greedy"): 

    Q = np.zeros((env.possible_states , 3))
    ep_rewards = []
    episodes_length = []
    state_counts = [0 for i in range(env.possible_states)]
    for episode_index in range(num_episodes):
        if (episode_index % 1000 == 0):
            print(str(episode_index) + "/" + str(num_episodes), learning_policy)
        state = env.reset()
        state = env.discretize_position(state)
        state_index = env.compute_state_index(state)
        # print(Q[state_index, :])

        done = False
        reward_sum = 0
        n_steps = 0
        while not done:  
            n_steps += 1
            # Choose action       
            if learning_policy == "greedy":
                epsilon = 0
                action, _ = egreedy_policy(Q, state_index, epsilon, state_counts = state_counts)
            elif learning_policy == "constant-epsilon":
                epsilon = exploration_rate
                action, _ = egreedy_policy(Q, state_index, epsilon, state_counts = state_counts)
            elif learning_policy == "epsilon-greedy": 
                epsilon = exploration_rate/max(state_counts[state_index], 1)
                action, state_counts = egreedy_policy(Q, state_index, epsilon, state_counts)
            elif learning_policy == "boltzmann-exploration":
                C = exploration_rate
                action, state_counts = boltzmann_exploration(Q, state_index, state_counts, C)
            
            
            next_state, reward, done, _ = env.step(action)
            next_state = env.discretize_position(next_state)
            next_state_index = env.compute_state_index(next_state)
            reward_sum += reward

            # Update q_values      
            try: 
                td_target = reward + discount_factor * np.max(Q[next_state_index, :])   # target found using q-learning
            except:
                print(next_state, next_state_index)
            td_error = td_target - Q[state_index, action]
            Q[state_index, action] += learning_rate * td_error
            
            # Update state
            state = next_state
            state_index = next_state_index

            if n_steps >= env._max_episodes_steps:
                done = True
            
            
        episodes_length.append(n_steps)
        ep_rewards.append(reward_sum)
    
    return ep_rewards, Q, episodes_length



def wrapper_parallel_computing(learning_policy = "epsilon-greedy", simulation_index = 0, exploration_rate = 0, n_episodes = 10000):
    env = MountainCarEnv(digits = 2, simulation_index=simulation_index)  
    np.random.seed(simulation_index)
    rewards_sum, Q, episode_length = q_learning(env, n_episodes, exploration_rate = exploration_rate, learning_rate=0.1, discount_factor= 0.9, learning_policy = learning_policy)
        
    return episode_length

num_episodes = 10000
n_simulations = 10
exploration_types = ["greedy", "constant-epsilon", "epsilon-greedy", "boltzmann-exploration"]
exploration_rates =  [0, 0.1, 0.25, 1.5]
optimal_exploration = zip(exploration_types, exploration_rates)
lista_possibili_gruppi_di_parametri = [(exploration_type, simulation_index, exploration_rate) for exploration_type, exploration_rate in optimal_exploration for simulation_index in range(n_simulations)]
print(lista_possibili_gruppi_di_parametri)

pool = mp.Pool( mp.cpu_count())
results =  pool.starmap(wrapper_parallel_computing, [(exploration_type, simulation_index, exploration_rate, num_episodes) for exploration_type, simulation_index, exploration_rate in lista_possibili_gruppi_di_parametri])  # , callback= collect_result))

results_dataframe = pd.DataFrame(np.zeros((num_episodes, len(exploration_types)*1))) #
for learning_policy_index in range( len(exploration_types) ): #
    average_values = [0 for i in range(num_episodes)]
    for episode_index in range(num_episodes):
        average_value = np.zeros((n_simulations, 1))
        for simulation_index in range(n_simulations):
            average_value[simulation_index] = results[learning_policy_index * n_simulations + simulation_index][episode_index]
        average_values[episode_index] = np.mean(average_value)
    results_dataframe.iloc[:, learning_policy_index] = average_values 

episodes_considered = 5
sum_methods = np.zeros((len(exploration_types)))
sd_methods = np.zeros((len(exploration_types)))
mean_variance_dataframe = pd.DataFrame(np.zeros((len(exploration_types), 2))) 
mean_variance_dataframe.columns = ["mean", "Standard deviation"]
mean_variance_dataframe.index = exploration_types

for learning_policy_index in range( len(exploration_types) ):
    values_to_consider = np.zeros((n_simulations, episodes_considered))
    for episode_index in range(num_episodes- episodes_considered, num_episodes):
        average_value = np.zeros((n_simulations, 1))
        for simulation_index in range(n_simulations):
            values_to_consider[simulation_index, episode_index - num_episodes + episodes_considered] = results[learning_policy_index * n_simulations + simulation_index][episode_index]
    print(exploration_types[learning_policy_index])
    print(values_to_consider)
    sums = np.sum(values_to_consider, axis = 1) 
    print(sums)
    sum_methods[learning_policy_index] = np.mean(sums)
    print(np.mean(sums))
    
    sd_methods[learning_policy_index] = math.sqrt(np.var(sums))
    print(math.sqrt(np.var(sums)))
    
mean_variance_dataframe.iloc[:, 0] = sum_methods
mean_variance_dataframe.iloc[:, 1] = sd_methods
print(mean_variance_dataframe)  # computations of average and variance of each method

print(mean_variance_dataframe)

results_dataframe.columns = exploration_types
mean_variance_dataframe.to_csv("data/mean_variance_undirected_mountain_car.csv", index = False)
results_dataframe.to_csv("data/comparison_rewards_undirected_mountain_car.csv", index = False)

