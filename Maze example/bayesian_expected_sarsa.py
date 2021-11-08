from operator import mul, pos
import gym
import numpy as np # 1. Load Environment and Q-table structure
import math
from numpy.core.fromnumeric import mean 
import pandas as pd
import matplotlib.pyplot as plt
from scipy.special import softmax
from scipy.integrate import quadrature, quad, tplquad
from scipy.special import gamma, digamma
from scipy.special import beta as beta_function
from scipy.stats import t as t_function

from gym import spaces
from gym.utils import seeding

import multiprocessing as mp

from mazeEnv import Maze


def change_range(values, vmin=0, vmax=1):
    start_zero = values - np.min(values)
    return (start_zero / (np.max(start_zero) + 1e-7)) * (vmax - vmin) + vmin


def greedy_selection(Q, state, state_counts, exploration_rate = 1, possible_actions = 2):
    # we simply take the action with the highest expected value. Does not take account of the uncertainty about the Q-value
    # print(Q[state, :, 0])
    # we define a boltzmann distribution based on the value of Q[state, :, 0]
    state_counts[state] += 1
    if np.max(Q[state, :, 0]) == np.min(Q[state, :, 0]) and np.max(Q[state, :, 1]) == np.min(Q[state, :, 1]) and np.max(Q[state, :, 2]) == np.min(Q[state, :, 2]) and np.max(Q[state, :, 3]) == np.min(Q[state, :, 3]):
        return np.random.choice(possible_actions), np.ones((possible_actions, )) *(1/possible_actions), state_counts
    else:
        C = exploration_rate*(np.max(abs(Q[state, :, 0])) - np.min(abs(Q[state, :, 0])))
        if C==0:
            C = 1
        probabilites_vector = softmax([(math.log(state_counts[state])/C) *  Q[state, i, 0]      for i in range(possible_actions)])

        # now we need to draw an action according to the probabilty given by probabilites_vector
        action = np.random.choice(np.arange(0, possible_actions), p = probabilites_vector)
        return action, probabilites_vector, state_counts


def q_value_sampling_selection(Q, state_index, possible_actions):
    if np.max(Q[state_index, :, 0]) == np.min(Q[state_index, :, 0]) and np.max(Q[state_index, :, 1]) == np.min(Q[state_index, :, 1]) and np.max(Q[state_index, :, 2]) == np.min(Q[state_index, :, 2]) and np.max(Q[state_index, :, 3]) == np.min(Q[state_index, :, 3]) :
        return np.random.choice(possible_actions), np.ones((possible_actions, ))*0.25
    else:
        # we use sampling to find the probabilities
        samples_number = 100
        occurrencies = np.zeros((possible_actions, ))
        
        samples = np.zeros((possible_actions, samples_number))
        for i in range(possible_actions):
            mu = Q[state_index, i, 0]
            lamb = Q[state_index, i, 1]
            alph = Q[state_index, i, 2]
            beta = Q[state_index, i, 3]
            samples[i, :] = np.random.standard_t(2*alph, samples_number) * np.sqrt(beta / (alph * lamb )) + mu
        
        for sample_index in range(samples_number):
            occurrencies[np.argmax(samples[:, sample_index])] += 1

        occurrencies = occurrencies/samples_number
        return np.random.choice(possible_actions, size = 1, p = occurrencies )[0], occurrencies
        

def compute_c(q_values, state, action):
    # numerator = q_values[state, action, 2] * gamma(q_values[state, action, 2] + 1/2) * math.sqrt(q_values[state, action, 3])
    #print(q_values[state, action, 3])
    numerator = q_values[state, action, 2]  * math.sqrt(abs(q_values[state, action, 3]))

    # denominator = (q_values[state, action, 2] - 0.5) * gamma(q_values[state, action, 2]) * gamma(1/2) * q_values[state, action, 2] * math.sqrt(2 * q_values[state, action, 1])
    denominator = (q_values[state, action, 2] - 0.5) * beta_function(q_values[state, action, 2], 1/2) * q_values[state, action, 2] * math.sqrt(2 * q_values[state, action, 1])
    
    # print(denominator)
    third_term = 1 + ( q_values[state, action, 0]**2 ) / ( 2 * q_values[state, action, 2] )

    return (numerator/denominator) * third_term ** (-(q_values[state, action, 2] + 1/2))


def find_best_actions(q_values, state):
    best_action = np.argmax(q_values[state, :, 0])
    second_best_action = np.argpartition(q_values[state, : ,0], -2)[-2]

    return best_action, second_best_action

def myopic_vpi_selection(q_values = 0, state = 0, possible_actions = 4, state_counts = 0):
    state_counts[state] += 1
    if np.max(q_values[state, :, 0]) == np.min(q_values[state, :, 0]) and np.max(q_values[state, :, 1]) == np.min(q_values[state, :, 1]) and np.max(q_values[state, :, 2]) == np.min(q_values[state, :, 2]) and np.max(q_values[state, :, 3]) == np.min(q_values[state, :, 3]):
        # print("parametri uguali")
        return np.random.choice(possible_actions), np.ones((possible_actions, )) * 1/possible_actions, state_counts
    else:
        # print("parametri diversi")
        vpi = np.zeros((1, possible_actions))
        # we have only two possible actions in this case (otherwise ot might be a good idea to define a function that finds the two best actions)
        
        best_action, second_best_action = find_best_actions(q_values, state)

        for action in range(possible_actions):
            c = compute_c(q_values, state, action)
            c = max(c, 10**(-15))
            # print(c)
            if best_action == action:
                vpi[0, action] = c + (q_values[state, second_best_action, 0] - q_values[state, action, 0]) * t_function.cdf( (q_values[state, second_best_action , 0] - q_values[state, action, 0])*(q_values[state, action, 1] * q_values[state, action, 2]/q_values[state, action, 3])**1/2  , 2 * q_values[state, action, 2])
            else:
                vpi[0, action] = c + (q_values[state, action, 0] - q_values[state, best_action, 0]) *( 1 - t_function.cdf( (q_values[state, best_action, 0] - q_values[state, action, 0])*(q_values[state, action, 1] * q_values[state, action, 2] /q_values[state, action, 3])**1/2  , 2 * q_values[state, action, 2] ) )

        if np.max(q_values[state, :, 0] + vpi[0, :]) == np.min(q_values[state, :, 0] + vpi[0, :]) :
            return np.random.choice(possible_actions), np.ones((possible_actions, )) * 1/possible_actions, state_counts
        else:
            exploration_rate = 1
            C = exploration_rate*(np.max(q_values[state, :, 0] + vpi[0, :]) - np.min(q_values[state, :, 0] + vpi[0, :]) )
            C = max(C, 0.1)
            probabilites_vector = softmax([(math.log(state_counts[state])/C) *  (q_values[state, i, 0] + vpi[0, i])   for i in range(4)])
            probabilites_vector = np.ones((possible_actions, )) * probabilites_vector
            return np.random.choice(possible_actions, size = 1, p = probabilites_vector )[0], probabilites_vector, state_counts

def doppio_prodotto(action_probability, Q, possible_actions = -1, next_state = -1):
    if possible_actions <= 1:
        return 0
    # print(action_probability)
    sum_terms = 0
    for i in range(possible_actions):
        for j in range(i+1, possible_actions):
            sum_terms += action_probability[next_state, i]*action_probability[next_state, j] * Q[next_state, i, 0] * Q[next_state, j, 0]
    return sum_terms

def dot(K, L):
   if len(K) != len(L):
      return 0

   return sum(i[0] * i[1] for i in zip(K, L))


def square_vector(K):
    return [K[i]**2  for i in range(len(K))]

def bayesian_n_step_expected_sarsa(env, num_episodes=500,  discount_factor=0.99, action_selection = "", values_update = "", n_steps = -1):  # buoni risultati: [0, .1, 1.001, 1] , [0, .1, 1.1, .25]

    num_states = 49*8
    if action_selection == "greedy_selection":
        Q = np.ones((env.num_states , 4, 4)) * [0, .25, 1.0001, .25]  
        n_steps = 1  
    elif action_selection == "q_value_sampling":
        Q = np.ones((env.num_states , 4, 4)) * [0, 1, 5, .1]
        n_steps = 1
    elif action_selection == "myopic_vpi_selection":
        Q = np.ones((env.num_states , 4, 4)) * [0, .25, 1.25, 1]
        n_steps = 1 # for each (state, action) pair we store the parameters of the corresponding distribution
    ep_rewards = []
    possible_actions = 4
    episodes_length = []
    action_probability = np.ones((num_states, 4)) * 0.25
    state_counts = np.zeros((num_states, ))

    for episode_index in range(num_episodes):
        if (episode_index % 100 == 0):
            print(str(episode_index) + "/" + str(num_episodes) + " " + action_selection + " " + values_update)
        
        future_states = []
        future_rewards = []
        future_actions = []
       

        t = 0
        tau = t - n_steps + 1
        terminal_time = math.inf

        state = env.reset()  
        future_states.append(state)   

        # 
        
        # future_actions.append(action)
        # we add R_0 = 0 to the list of rewards
        future_rewards.append(-math.inf)
        done = False
        reward_sum = 0

        while tau < terminal_time -1:     
            # print(t, tau, terminal_time)
            if (t < terminal_time):
                # take action A_t
                # action = future_actions[t]
                state = future_states[t]

                if action_selection  == "greedy_selection":
                    action, action_probabilities, state_counts = greedy_selection(Q, state, state_counts, exploration_rate = 1, possible_actions= possible_actions) 
                elif action_selection == "q_value_sampling":
                    action, action_probabilities = q_value_sampling_selection(Q, state, possible_actions = possible_actions)
                elif action_selection == "myopic_vpi_selection":
                    action, action_probabilities, state_counts = myopic_vpi_selection(Q, state, possible_actions = possible_actions, state_counts = state_counts)

                action_probability[state, :] = action_probabilities

                future_actions.append(action)
                
                
                
                next_state, reward, done = env.step(action)

                # observe and store the next reward as R_{t+1} and the next state as S_{t+1}
                # indeed thay are going to be element t+1 in the respective lists
                future_states.append(next_state)
                reward_sum += reward
                future_rewards.append(reward)   

                if t >= env.max_steps_episode:
                    done = True
                # If S_{t+1} is terminal, then update terminal_time
                if done:
                    terminal_time = t+1
                    # final_reward = reward
                    # print("terminal time = ", terminal_time)
                    
            tau = t - n_steps + 1
            if tau >= 0:
                # we update the values of the Q function of the corresponding state and action at time tau (MOMENT UPDATING)
                gamma_list = [discount_factor ** i for i in range(min(n_steps, terminal_time - tau ))]
                relevant_rewards = [future_rewards[i] for i in range(tau+1 , min(tau + n_steps +1 , terminal_time +1))]   # arriva fino a tau+n_steps
                if min(tau + n_steps , terminal_time) == tau+n_steps :
                    expected_value = dot( action_probability[future_states[tau + n_steps]]  , Q[future_states[tau + n_steps], :, 0]  )
                    M1 = dot(relevant_rewards, gamma_list) + discount_factor **  n_steps * expected_value
                    square = [ (Q[future_states[tau + n_steps], i, 1] +1 )/(Q[future_states[tau + n_steps], i, 1]) * (Q[future_states[tau + n_steps], i, 3])/(Q[future_states[tau + n_steps], i, 2] -1) + Q[future_states[tau + n_steps], i, 0]**2   for i in range(possible_actions) ] 
                    M2 = (dot(relevant_rewards, gamma_list))**2 + 2 * dot(relevant_rewards, gamma_list) * expected_value + discount_factor**2 * dot( square_vector(action_probability[future_states[tau + n_steps]]) , square ) + 2 * discount_factor**2 * doppio_prodotto(action_probability, Q, possible_actions = possible_actions, next_state= next_state)
                else: # there is no part dedicated to the expected reward at the last step, as the episode has ended
                    M1 = dot(relevant_rewards, gamma_list)
                    M2 = (dot(relevant_rewards, gamma_list))**2

                mu_0 = Q[future_states[tau], future_actions[tau], 0]
                lambda_0 = Q[future_states[tau], future_actions[tau], 1]
                alpha_0 = Q[future_states[tau], future_actions[tau], 2]
                beta_0 = Q[future_states[tau], future_actions[tau], 3]

                mu_1 = (lambda_0 * mu_0 + 1 * M1)/(lambda_0 + 1)
                lambda_1 = lambda_0 + 1
                alpha_1 = alpha_0 + 1/2
                beta_1 = beta_0 + 1/2 * 1 * (M2 - M1**2) + ( lambda_0 * (M1 - mu_0)**2 )/(2 * (lambda_0 + 1))

                Q[future_states[tau], future_actions[tau], 0] = mu_1
                Q[future_states[tau], future_actions[tau], 1] = lambda_1
                Q[future_states[tau], future_actions[tau], 2] = alpha_1
                Q[future_states[tau], future_actions[tau], 3] = beta_1
            t += 1
        
        print(env.flags_visited)
        ep_rewards.append(reward_sum)

    
    return ep_rewards, Q


def wrapper_parallel_computing(action_selection = "", values_update = "", simulation_index = 0, n_episodes = 1000, n_steps = 5):
    env = Maze()
    np.random.seed(simulation_index)
    rewards_sum, Q = bayesian_n_step_expected_sarsa(env, num_episodes= n_episodes, discount_factor= 0.99, action_selection = action_selection, values_update = values_update, n_steps = n_steps)
    
    return rewards_sum




# env.observation.n, env.action_space.n gives number of states and action in env loaded# 2. Parameters of Q-learning

n_simulations = 7
action_selection_types = ["q_value_sampling"] # ["greedy_selection", "q_value_sampling", "myopic_vpi_selection" ] 
q_values_updating_types = ["moment_updating"] 
lista_possibili_metodi = [(action_selection_method, q_value_updating_method) for action_selection_method in action_selection_types for q_value_updating_method in q_values_updating_types ]

lista_possibili_gruppi_di_parametri = [(action_selection_method, q_value_updating_method, simulation_index) for action_selection_method in action_selection_types for q_value_updating_method in q_values_updating_types for simulation_index in range(n_simulations)  ]
print(lista_possibili_gruppi_di_parametri)

pool = mp.Pool( mp.cpu_count()-1 )

relevant_Q_tables = []

n_steps = 1
n_episodes = 10000
results =  pool.starmap(wrapper_parallel_computing, [(action_selection_method, q_value_updating_method, simulation_index, n_episodes, n_steps) for action_selection_method, q_value_updating_method, simulation_index in lista_possibili_gruppi_di_parametri])  # , callback= collect_result))

results_dataframe = pd.DataFrame(np.zeros((n_episodes, len(action_selection_types)*len(q_values_updating_types)))) #

for learning_policy_index in range( len(lista_possibili_metodi) ): #
    average_values = [0 for i in range(n_episodes)]
    for episode_index in range(n_episodes):
        average_value = np.zeros((n_simulations, 1))
        for simulation_index in range(n_simulations):
            average_value[simulation_index] = results[learning_policy_index * n_simulations + simulation_index][episode_index]
        average_values[episode_index] = np.mean(average_value)
    results_dataframe.iloc[:, learning_policy_index] = average_values 

episodes_considered = 5
sum_methods = np.zeros((len(lista_possibili_metodi)))
sd_methods = np.zeros((len(lista_possibili_metodi)))
mean_variance_dataframe = pd.DataFrame(np.zeros((len(action_selection_types), 2)))
mean_variance_dataframe.columns = ["mean", "Standard deviation"]
# try:
mean_variance_dataframe.index = lista_possibili_metodi
# except:
#     mean_variance_dataframe.index = lista_possibili_metodi + ["eee", "lll"]

for learning_policy_index in range( len(lista_possibili_metodi) ):
    values_to_consider = np.zeros((n_simulations, episodes_considered))
    for episode_index in range(n_episodes- episodes_considered, n_episodes):
        average_value = np.zeros((n_simulations, 1))
        for simulation_index in range(n_simulations):
            values_to_consider[simulation_index, episode_index - n_episodes + episodes_considered] = results[learning_policy_index * n_simulations + simulation_index][episode_index]
    print(lista_possibili_metodi[learning_policy_index])
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


results_dataframe.columns = lista_possibili_metodi
mean_variance_dataframe.to_csv("data/mean_variance_rewards_bayesian_expected_sarsa_final.csv", index = False)
results_dataframe.to_csv("data/comparison_rewards_bayesian_step_expected_sarsa_final.csv", index = False)

for i in range(len(lista_possibili_metodi)):
    # plt.plot(results_dataframe.iloc[:, i])
    plt.plot(pd.Series(results_dataframe.iloc[:, i]).rolling(100, min_periods = 10).mean() )
plt.ylim([0, 100])
plt.rcParams.update({'font.size': 12})
plt.xlabel("Number of episodes", fontsize = 12)
plt.ylabel("Reward", fontsize = 12)
plt.legend(lista_possibili_metodi)
plt.show()
 