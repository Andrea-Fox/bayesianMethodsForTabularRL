from os import error
import random
from typing import Optional
import numpy as np
from numpy.core.defchararray import index
from numpy.core.fromnumeric import mean
from numpy.lib.arraysetops import isin
import pandas as pd
import math

from scipy.special import softmax
from mazeEnv import Maze
import matplotlib.pyplot as plt

import multiprocessing as mp


def change_range(values, vmin=0, vmax=1):
    start_zero = values - np.min(values)
    return (start_zero / (np.max(start_zero) + 1e-7)) * (vmax - vmin) + vmin
  

def boltzmann_exploration(q_values, state, state_counts, exploration_rate, possible_actions = -1):
    # we choose C_t equal to the maximum between the values

    if state_counts[state] == 0:
        state_counts[state] += 1
        return np.random.choice(possible_actions), state_counts, np.ones((possible_actions, ))*(1/possible_actions)
    else:
        C = exploration_rate*max(abs(q_values[state, 0] - q_values[state, 1]), 0.1)
        if C==0:
            C = 1

        state_counts[state] += 1
        probabilities_vector = softmax([(math.log(state_counts[state])/C) *  q_values[state, i]      for i in range(possible_actions)])
        # now we need to draw an action according to the probabilty given by probabilites_vector

        action = np.random.choice(np.arange(0, possible_actions), p = probabilities_vector)

        return action, state_counts, probabilities_vector

    

def egreedy_policy(q_values, state, state_counts, epsilon = 0.1, possible_actions = -1):
    ''' 
    Choose an action based on a epsilon greedy policy.    
    A random action is selected with epsilon probability, else select the best action.    
    '''
    state_counts[state] += 1
    if state_counts[state] == 1:
        return np.random.choice(possible_actions), state_counts, np.ones((possible_actions, ))*(1/possible_actions)
    else:
        # at first we find the optimal action
        best_action = np.argmax(q_values[state, :])
        probability = np.ones((possible_actions, )) * epsilon/possible_actions
        probability[best_action] = 1 - epsilon + epsilon/possible_actions

        return np.random.choice(np.arange(0, possible_actions), p = probability), state_counts, probability
        # if np.random.random() < epsilon:
        #     return np.random.choice(4), state_counts
        # else:
        #     # print(np.argmax(q_values[state, ]))
        #     return np.argmax(q_values[state, :]), state_counts

# definiamo il prodotto scalare tra liste
def dot(K, L):
   if len(K) != len(L):
      return 0

   return sum(i[0] * i[1] for i in zip(K, L))

def compute_sigma(sigma, time):
    if isinstance(sigma, int):
        if sigma == 1:
            return 1
        elif sigma == 0:
            return 0
        else:
            # trovare come definire che è un errore
            return -1
    
    elif isinstance(sigma, float):
        if sigma <1 and sigma > 0:
            return sigma
        else: 
            # trovare come definire che è un errore
            return -1
    
    elif isinstance(sigma, str):
        if sigma == "dynamic":
            # qui andrebbe definita la funzione dinamica, come fatto nell'articolo
            return 0.95**time
        else:
            # trovare come definire che è un errore
            return -1
    else: 
        # trovare come definire che è un errore
        return -1
        
    


# we need to define the n-step method
def q_sigma(env, num_episodes=500, learning_rate=0.5, discount_factor=1, exploration_rate = 0.1, n = 5, num_states = 50, sigma = -1, learning_policy = ""):
    #print("n_step sarsa ", learning_rate, " ", n)
    ep_rewards = []
    terminal_times = []
    possible_actions = 4 
    Q = np.zeros((49*8, possible_actions))  

    probabilities = np.ones((49*8, possible_actions))*(1/possible_actions)

    state_counts = np.zeros((49*8, 1))
    # print(value_function)
    for index_episode in range(num_episodes):
        
        future_states = []
        future_rewards = []
        future_delta = []
        future_actions = []

        t = 0
        tau = t - n + 1
        terminal_time = math.inf
        

        state = env.reset()
        # we add S_0 to the list of states
        future_states.append(state)   

        if learning_policy == "greedy": 
            action, state_counts, probability_vector = egreedy_policy(Q, state, state_counts, 0.0, possible_actions = possible_actions)
        elif learning_policy == "constant-epsilon":
            action, state_counts, probability_vector = egreedy_policy(Q, state, state_counts, exploration_rate, possible_actions = possible_actions)
        elif learning_policy == "epsilon_greedy": 
            epsilon = exploration_rate / (state_counts[state]+1)
            action, state_counts, probability_vector = egreedy_policy(Q, state, state_counts, epsilon, possible_actions = possible_actions)
        elif learning_policy == "boltzmann_exploration":
            action, state_counts, probability_vector = boltzmann_exploration(Q, state, state_counts, exploration_rate, possible_actions = possible_actions)
        probabilities[state, :] = probability_vector
        
        future_actions.append(action)
        # we add R_0 = 0 to the list of rewards
        future_rewards.append(0)
        done = False
        reward_sum = 0

        while tau != terminal_time -1 :   

            if (t < terminal_time):
                # take action A_t
                action = future_actions[t]
                state = future_states[t]
                next_state, reward, done = env.step(action)

                # observe and store the next reward as R_{t+1} and the next state as S_{t+1}
                # indeed thay are going to be element t+1 in the respective lists
                future_states.append(next_state)
                reward_sum += reward
                future_rewards.append(reward)    
                if t>= env.max_steps_episode:
                    done = True
                # If S_{t+1} is terminal, then update terminal_time
                if done:
                    terminal_time = t+1
                    # final_reward = reward
                    # print("terminal time = ", terminal_time)
                    delta = reward - Q[state, action]
                    future_delta.append(delta)

                else:
                    if learning_policy == "greedy": 
                        next_action, state_counts, probability_vector = egreedy_policy(Q, next_state, state_counts, 0.0, possible_actions = 4)
                    elif learning_policy == "constant-epsilon":
                        epsilon = exploration_rate
                        next_action, state_counts, probability_vector = egreedy_policy(Q, next_state, state_counts, exploration_rate, possible_actions = 4)
                    elif learning_policy == "epsilon_greedy": 
                        epsilon = exploration_rate/(state_counts[state]+1)
                        next_action, state_counts, probability_vector = egreedy_policy(Q, next_state, state_counts, exploration_rate, possible_actions = 4)
                    elif learning_policy == "boltzmann_exploration":
                        next_action, state_counts, probability_vector = boltzmann_exploration(Q, next_state, state_counts, exploration_rate, possible_actions = 4)
                    else:
                        print(learning_policy)

                    probabilities[next_state, :] = probability_vector
                    future_actions.append(next_action)

                    expected_value = dot(probability_vector, Q[next_state, :])
                    
                    
                    # delta = reward  + value_function[next_state] - value_function[state]
                    delta = reward + discount_factor * ( compute_sigma(sigma, index_episode+1) * Q[next_state, next_action] + (1-compute_sigma(sigma, index_episode+1)) * expected_value ) - Q[state, action]
                    future_delta.append(delta)
                          
            tau = t - n + 1
            if tau >= 0:
                E = 1
                # return_G = value_function[ future_states[tau] ]
                return_G = Q[ future_states[tau] , future_actions[tau]]
                final_index = min(terminal_time-1, tau + n -1) # il -1 non lo mettiamo così poi nel range alla riga sotto non dobbiamo aggiungere il +1                  
                for k in range(tau, final_index+1):
                    return_G += E * future_delta[k]
                    E *= discount_factor * ((1 - compute_sigma(sigma, index_episode)) * probabilities[future_states[k], future_actions[k]] + compute_sigma(sigma, index_episode) )    
                    # E è sempre 1 in questo caso, per come abbiamo scelto sigma, per come sono le percentuale e perché il discount factor è 1
                    # qui ci sarebbe l'aggiornamento dell'importance sampling ratio, che però è sempre 1
                    
                # value_function[ future_states[tau] ] += learning_rate * (final_return - value_function[ future_states[tau] ]) 
                Q[ future_states[tau], future_actions[tau] ] += learning_rate * (return_G  - Q[ future_states[tau], future_actions[tau]] )

            # state = next_state   
            t += 1
        # at the end of each episode we can compute the RMSE at this point
        # print(value_function[0:num_states, :])
        # print(rmse(value_function[0:num_states, :], optimal_values ))           
        ep_rewards.append(reward_sum) 
        terminal_times.append(terminal_time)
    return ep_rewards, Q, terminal_times
    # return value_function


def wrapper_parallel_computing(n_episodes= 500, learning_rate = 0.1, discount_factor = 1, n = 1, n_simulations = 50, num_states = 49*8, sigma = 0, exploration_rate = 0.1, learning_policy = ""):
    # print("num states= ", num_states)
    # print("learning rate =  ", learning_rate)
    # print("n = ", n)
    env = Maze()
    ep_rewards_list = [] 
    terminal_times_list = [] 
    average_reward = np.zeros((n_episodes, 1))  
    for simulation_index in range(n_simulations):
        np.random.seed(simulation_index)
        ep_rewards, Q, terminal_times = q_sigma(env, num_episodes = n_episodes, learning_rate = learning_rate,  discount_factor = discount_factor, exploration_rate= exploration_rate, n = n, num_states = num_states, sigma = sigma, learning_policy= learning_policy)
        ep_rewards_list.append(ep_rewards)
        terminal_times_list.append(terminal_times)

    # print("fine")
    # now we compute the average reward of each episode
    for index_episode in range(n_episodes):
        values = np.zeros((n_simulations, 1))
        for index_simulation in range(n_simulations):
            values[index_simulation] = terminal_times_list[index_simulation][index_episode]
        average_reward[index_episode] = np.mean(values)
    
    # print("risultato = ", risultato)

    episodes_considered = 5
    sum_methods = 0
    sd_methods = 0
    mean_variance_dataframe = pd.DataFrame(np.zeros((1, 2)))  #CAMBIARE!!!
    mean_variance_dataframe.columns = ["mean", "Standard deviation"]

    
    values_to_consider = np.zeros((n_simulations, episodes_considered))
    for episode_index in range(n_episodes- episodes_considered, n_episodes):
        average_value = np.zeros((n_simulations, 1))
        for simulation_index in range(n_simulations):
            values_to_consider[simulation_index, episode_index - n_episodes + episodes_considered] = terminal_times_list[simulation_index][episode_index]
    # print(values_to_consider)
    sums = np.sum(values_to_consider, axis = 1) 
    # print(sums)
    sum_methods = np.mean(sums)
    sd_methods = math.sqrt(np.var(sums))
    
    mean_variance_dataframe.iloc[0, 0] = sum_methods
    mean_variance_dataframe.iloc[0, 1] = sd_methods
    # print("risultato = ", risultato)
    print(learning_policy, sigma)
    print(mean_variance_dataframe)
    
    print( [num_states, n, learning_rate, n_episodes, sigma, exploration_rate, learning_policy])
    return average_reward

# tutte le altre cose per inizializzare

print("Number of processors: ", mp.cpu_count())

# per sicurezza lasciamo un core libero che non si sa mai con tutti che vanno al 100%
pool = mp.Pool( mp.cpu_count())




n_episodes = 100

n_simulations = 10

n_values = [1, 2, 5, 10, 20, 50]  
alpha_values = 0.1 # general optimal value of the learning rate  
num_states = 49*8
learning_policies = ["greedy", "constant-epsilon", "epsilon_greedy", "boltzmann_exploration"]
exploration_rates = [0, 0.2, 0.25, 1.5]      #optimal exploration rates in the case n=1
learning_policies_with_parameters = list(zip(learning_policies, exploration_rates))
print(learning_policies_with_parameters)
sigma_values = [0, 0.25, 0.5, 0.75, 1] #, "dynamic"]


# lista_possibili_gruppi_di_parametri = [(sigma, n_episodes, learning_policy, exploration_rate, n )  for sigma in sigma_values  for learning_policy, exploration_rate in learning_policies_with_parameters for n in n_values]
lista_possibili_gruppi_di_parametri = [(1, n_episodes, "constant-epsilon", .1, 1), (1, n_episodes, "greedy", 0, 5), (0.5, n_episodes, "epsilon_greedy", .1, 50), (0.25, n_episodes, "constant-epsilon", 0.1, 5), (0.25, n_episodes, "boltzmann_exploration", 1.5, 5) ]

print(lista_possibili_gruppi_di_parametri)

results = pool.starmap(wrapper_parallel_computing, [(n_episodes, alpha_values, 0.99,  n, n_simulations , num_states, sigma, exploration_rate, learning_policy) for sigma, n_episodes, learning_policy, exploration_rate, n  in lista_possibili_gruppi_di_parametri])  # , callback= collect_result))
pool.close()
print(results)
dataframe_values = pd.DataFrame(np.zeros((n_episodes, len(lista_possibili_gruppi_di_parametri))))

for i in range(len(lista_possibili_gruppi_di_parametri)):
    dataframe_values.iloc[:, i] = results[i]

print(dataframe_values)
dataframe_values.columns = lista_possibili_gruppi_di_parametri
dataframe_values.to_csv("data/maze_q_sigma_optimal_values.csv", index = False)

