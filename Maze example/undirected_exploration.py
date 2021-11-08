from operator import mul
from types import resolve_bases
import gym
import numpy as np # 1. Load Environment and Q-table structure
import math 
import pandas as pd
import matplotlib.pyplot as plt
from pandas.io.stata import stata_epoch
from matplotlib.colors import hsv_to_rgb

from scipy.special import softmax

from gym import spaces
from gym.utils import seeding

import multiprocessing as mp

from mazeEnv import Maze

def change_range(values, vmin=0, vmax=1):
    start_zero = values - np.min(values)
    return (start_zero / (np.max(start_zero) + 1e-7)) * (vmax - vmin) + vmin

   

def boltzmann_exploration(q_values, state, state_counts, exploration_rate, n_steps = -1):
    # we choose C_t equal to the maximum between the values

    
    if np.sum(state_counts) == 0:
        state_counts[state] += 1
        return np.random.choice(4), state_counts
    else:
        C = exploration_rate*max(abs(q_values[state, 0] - q_values[state, 1]), 0.1)
        if C==0:
            C = 1

        state_counts[state] += 1
        probabilites_vector = softmax([(math.log(state_counts[state])/C) *  q_values[state, i]      for i in range(4)])

        # now we need to draw an action according to the probabilty given by probabilites_vector
        action = np.random.choice(np.arange(0, 4), p = probabilites_vector)
        return action, state_counts

    

def egreedy_policy(q_values, state, state_counts, epsilon = 0.1):
    ''' 
    Choose an action based on a epsilon greedy policy.    
    A random action is selected with epsilon probability, else select the best action.    
    '''
    

    if np.sum(state_counts) == 0:
        state_counts[state] += 1
        return np.random.choice(4), state_counts
    else:
        state_counts[state] += 1

        if np.random.random() < epsilon:
            return np.random.choice(4), state_counts
        else:
            # print(np.argmax(q_values[state, ]))
            action = np.argmax(q_values[state, :])
            
            return  action, state_counts

def q_learning(env, num_episodes=500, exploration_rate=0.1, learning_rate=0.1, discount_factor=0.9, learning_policy = "epsilon-greedy"): 
    
    Q = np.zeros((49*8 , 4))
    ep_rewards = []
    state_counts = np.zeros((49*8,  1))


    for episode_index in range(num_episodes):

        

        if (episode_index % 100 == 0):
            print(str(episode_index) + "/" + str(num_episodes) + " " + learning_policy)
        state = env.reset()  
        # print("stato iniziale = ", state)  

        
        # print(Q[state_index, :])

        done = False
        reward_sum = 0
        n_steps = 0
        while not done:  
            n_steps += 1
            # Choose action       
            if learning_policy == "greedy": 
                action, state_counts = egreedy_policy(Q, state, state_counts, 0.0)
            elif learning_policy == "constant-epsilon":
                epsilon = exploration_rate
                action, state_counts = egreedy_policy(Q, state, state_counts, epsilon)
            elif learning_policy == "epsilon-greedy": 
                epsilon = exploration_rate/(state_counts[state]+1)
                action, state_counts = egreedy_policy(Q, state, state_counts, epsilon)
            elif learning_policy == "boltzmann-exploration":
                action, state_counts = boltzmann_exploration(Q, state, state_counts, exploration_rate)
            else:
                print(learning_policy)
            # Do the action
            # print(env.step(action))
            next_state, reward, done = env.step(action)
            reward_sum += reward

            # Update q_values      
            try: 
                td_target = reward + discount_factor * np.max(Q[next_state, :])   # target found using q-learning
            except:
                print(next_state, next_state)
            td_error = td_target - Q[state, action]
            Q[state, action] += learning_rate * td_error
            
            # Update state
            # if episode_index == 999:
            #     print(env._id_to_position(state))
            state = next_state
            
            if n_steps > env.max_steps_episode:
                done = True

            
        ep_rewards.append(reward_sum)
    
    return ep_rewards, Q

def play(q_values):
    env = Maze()
    state = env.reset()
    reward_sum = 0
    done = False
    n_steps = 0
    while not done:
        n_steps += 1    
        # Select action
        action = egreedy_policy(q_values, state, 0.0, episode_index == 0)
        # Do the action
        next_state, reward, done, _ = env.step(state, action)  
        if done:
            print(n_steps)
        reward_sum += reward
        # Update state and action        
        state = next_state  

        if n_steps > env._max_episodes_steps:
            done = True


    return reward_sum


def wrapper_parallel_computing(learning_policy = "epsilon-greedy", simulation_index = 0, exploration_rate = 0, n_episodes = 1000):
    env = Maze()
    np.random.seed(simulation_index)
    rewards_sum, Q = q_learning(env, n_episodes, exploration_rate = exploration_rate, learning_rate=0.1, discount_factor= 0.9, learning_policy = learning_policy )
    
    print(learning_policy)
    print(Q)
    print(".........................................................")
    # if simulation_index == 1000:
    #     relevant_Q_tables.append(Q)
    # elif simulation_index == 5000:
    #     relevant_Q_tables.append(Q)
    # elif simulation_index == 10000:
    #     relevant_Q_tables.append(Q)
    
    return rewards_sum


'''
- trovare il miglior parametro anche per espilon greedy !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

- mostrare come evolve la reward nel corso degli episodi, fino a 10000 (e quindi fare un 1000 simulazioni per ogni esempio)
- vedere come cambia la policy raccomondata dopo 1000, 5000, 10000 passi: usiamo la Q table ottenuta con il metodo e poi, attraverso l'approccio greedy vediamo quale è la qualità della policy che verrebbe raccomandata
  Confrontare sia quello che succede usando l'approccio greedy, che usando quello con i sample (nel caso dei metodi bayesian)
  Ripetere questa operazione (training + test) per 100 simulazioni e poi raccogliere i risultati numerici in una tabella (mostrare sia media che varianza a questo punto)
'''
n_simulations = 10
exploration_types = ["greedy", "constant-epsilon", "epsilon-greedy", "boltzmann-exploration"]

exploration_rates = [0,0.2, 0.25, 2.25]
optimal_couple_parameters = list(zip(exploration_types, exploration_rates))
print(optimal_couple_parameters)
lista_possibili_gruppi_di_parametri = [(exploration_type, simulation_index, exploration_rate) for exploration_type, exploration_rate in optimal_couple_parameters for simulation_index in range(n_simulations)  ]
print(lista_possibili_gruppi_di_parametri)

pool = mp.Pool( mp.cpu_count()-1 )

relevant_Q_tables = []
n_episodes = 10000
results =  pool.starmap(wrapper_parallel_computing, [(exploration_type, simulation_index, exploration_rate, n_episodes) for exploration_type, simulation_index, exploration_rate in lista_possibili_gruppi_di_parametri])  # , callback= collect_result))

print("fine")
# print(results)

# print on file the results, averaged for each episode
results_dataframe = pd.DataFrame(np.zeros((n_episodes, len(exploration_rates)))) #
for learning_policy_index in range( len(exploration_types) ): #
    average_values = [0 for i in range(n_episodes)]
    for episode_index in range(n_episodes):
        average_value = np.zeros((n_simulations, 1))
        for simulation_index in range(n_simulations):
            average_value[simulation_index] = results[learning_policy_index * n_simulations + simulation_index][episode_index]
        average_values[episode_index] = np.mean(average_value)
    results_dataframe.iloc[:, learning_policy_index] = average_values 

# computation of the averages for each episode of all the simulations:
episodes_considered = 5
sum_methods = np.zeros((len(exploration_types)))
sd_methods = np.zeros((len(exploration_types)))
mean_variance_dataframe = pd.DataFrame(np.zeros((len(exploration_types), 2)))  #CAMBIARE!!!
mean_variance_dataframe.columns = ["mean", "Standard deviation"]
mean_variance_dataframe.index = exploration_types

for learning_policy_index in range( len(exploration_types) ):
    values_to_consider = np.zeros((n_simulations, episodes_considered))
    for episode_index in range(n_episodes- episodes_considered, n_episodes):
        average_value = np.zeros((n_simulations, 1))
        for simulation_index in range(n_simulations):
            values_to_consider[simulation_index, episode_index - n_episodes + episodes_considered] = results[learning_policy_index * n_simulations + simulation_index][episode_index]
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


results_dataframe.columns = exploration_types

mean_variance_dataframe.to_csv("data/mean_variance_rewards_undirected_explorations.csv", index = False)
results_dataframe.columns = exploration_types
results_dataframe.to_csv("data/comparison_rewards_undirected_methods.csv", index = False)


print(results_dataframe)
for i in range(results_dataframe.shape[1]):
    plt.plot(pd.Series(results_dataframe.iloc[:, i]).rolling(100, min_periods = 10).mean()   )
plt.legend(exploration_types) #
plt.show()
