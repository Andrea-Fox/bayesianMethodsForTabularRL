from operator import mul
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


from chainEnv import Chain


def greedy_selection(q_values, state, n_steps, possible_actions = 2):
    # we simply take the action with the highest expected value. Does not take account of the uncertainty about the Q-value
    # print(Q[state, :, 0])
    if q_values[state, 0, 0] == q_values[state, 1, 0] and q_values[state, 0, 1] == q_values[state, 1, 1] and q_values[state, 0, 2] == q_values[state, 1, 2] and q_values[state, 0, 3] == q_values[state, 1, 3]:
        return np.random.choice(possible_actions)
    else:
        return np.argmax(q_values[state,: , 0])


def dot(K, L):
   if len(K) != len(L):
      return 0

   return sum(i[0] * i[1] for i in zip(K, L))

def bayesian_n_step_sarsa(env, num_episodes=500,  discount_factor=0.99, action_selection = "", values_update = "", n_steps = -1, initial_lamb = 0.1, initial_alpha = 1.1, initial_beta = 0.5):  # buoni risultati: [0, .1, 1.001, 1], [0, .1, 1.01, .05] , [0, 5, 1.01, 0.01]

    print(initial_lamb, initial_alpha, initial_beta, n_steps)
    if action_selection == "greedy_selection":
        Q = np.ones((env.num_states , 2, 4)) * [0, initial_lamb, initial_alpha, initial_beta]  
        n_steps = 1  
    ep_rewards = []
    episodes_length = []

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
                if action_selection == "greedy_selection":
                    action = greedy_selection(Q, state, t) 
                future_actions.append(action)
                
                
                
                next_state, reward, done = env.step(state, action)

                # observe and store the next reward as R_{t+1} and the next state as S_{t+1}
                # indeed thay are going to be element t+1 in the respective lists
                future_states.append(next_state)
                reward_sum += reward
                future_rewards.append(reward)    
                if t >= env.max_episodes_steps:
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
                    best_action = np.argmax(Q[future_states[tau + n_steps], :, 0])
                    M1 = dot(relevant_rewards, gamma_list) + discount_factor **  n_steps * Q[future_states[tau + n_steps], best_action, 0]
                    M2 = (dot(relevant_rewards, gamma_list))**2 + 2 * discount_factor ** n_steps * dot(relevant_rewards, gamma_list) * Q[future_states[tau + n_steps], best_action, 0] + (discount_factor ** (2*n_steps)) * ( ( (Q[future_states[tau + n_steps], best_action, 1] + 1)/(Q[future_states[tau + n_steps], best_action, 1]) )*(Q[future_states[tau + n_steps], best_action, 3]/(Q[future_states[tau + n_steps], best_action, 2]-1)) + (Q[future_states[tau + n_steps], best_action, 0])**2   )
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
           
        ep_rewards.append(reward_sum)
    
    return ep_rewards, Q


def wrapper_parallel_computing(action_selection = "", values_update = "", simulation_index = 0, lamb = .1 , alpha = 1.1, beta = .5, n_steps = -1, n_episodes = 1000):
    
    np.random.seed(simulation_index)
    env = Chain(possible_states)
    rewards_sum, Q = bayesian_n_step_sarsa(env, num_episodes= n_episodes, discount_factor= 0.99, action_selection = action_selection, values_update = values_update, n_steps = n_steps, initial_lamb = lamb, initial_alpha = alpha, initial_beta = beta)
    
    return rewards_sum




possible_states = 5

print("possible_states = ", possible_states)

state_counts = [0 for i in range(possible_states)]


# env.observation.n, env.action_space.n gives number of states and action in env loaded# 2. Parameters of Q-learning
n_simulations = 50
action_selection_types = ["greedy_selection"]
q_values_updating_types = ["moment_updating"] 
lista_possibili_metodi = [(action_selection_method, q_value_updating_method) for action_selection_method in action_selection_types for q_value_updating_method in q_values_updating_types ]

lambdas = [1, 0.01, .01, .001, .25, .1]
alphas = [2, 1.5, 1.0001, 2, 1.25, 1.001]
betas = [1.5, .25, .1, 3, .1, 2]
n_values = [1, 2, 3, 5, 10, 20]
optimal_parameters = list(zip(lambdas, alphas, betas, n_values))
print(optimal_parameters)
lista_possibili_gruppi_di_parametri = [(simulation_index, lamb, alpha, beta, n)  for simulation_index in range(n_simulations) for lamb, alpha, beta, n in optimal_parameters ]
print(lista_possibili_gruppi_di_parametri)

pool = mp.Pool(mp.cpu_count())

n_episodes = 1000
results =  pool.starmap(wrapper_parallel_computing, [("greedy_selection", "moment_updating", simulation_index, lamb, alpha, beta, n, n_episodes) for simulation_index, lamb, alpha, beta, n in lista_possibili_gruppi_di_parametri]) 
# print(results)
results_dataframe = pd.DataFrame(np.zeros((n_episodes, len(n_values)))) #
print(results_dataframe.shape)
for learning_policy_index in range( len(n_values) ): #
    average_values = [0 for i in range(n_episodes)]
    for episode_index in range(n_episodes):
        average_value = np.zeros((n_simulations, 1))
        for simulation_index in range(n_simulations):
            average_value[simulation_index] = results[learning_policy_index * n_simulations + simulation_index][episode_index]
        average_values[episode_index] = np.mean(average_value)
    results_dataframe.iloc[:, learning_policy_index] = average_values 

episodes_considered = 5
sum_methods = np.zeros((len(optimal_parameters)))
sd_methods = np.zeros((len(optimal_parameters)))
mean_variance_dataframe = pd.DataFrame(np.zeros((len(optimal_parameters), 2)))
mean_variance_dataframe.columns = ["mean", "Standard deviation"]
mean_variance_dataframe.index = n_values 
for learning_policy_index in range( len(optimal_parameters) ):
    values_to_consider = np.zeros((n_simulations, episodes_considered))
    for episode_index in range(n_episodes- episodes_considered, n_episodes):
        average_value = np.zeros((n_simulations, 1))
        for simulation_index in range(n_simulations):
            values_to_consider[simulation_index, episode_index - n_episodes + episodes_considered] = results[learning_policy_index * n_simulations + simulation_index][episode_index]
    # print(lista_possibili_metodi[learning_policy_index])
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


results_dataframe.columns = n_values
mean_variance_dataframe.to_csv("data/mean_variance_rewards_bayesian_q_learning_greedy.csv", index = False)
results_dataframe.to_csv("data/mean_variance_rewards_bayesian_q_learning_greedy.csv", index = False)





# results_dataframe = pd.read_csv("/home/andrea/Desktop/tesiReinforcmentLearning/codice/chain/risultati_numerici/comparison_rewards_bayesian_n_step.csv")

print(results_dataframe)

for i in range(len(n_values)):
    # plt.plot(results_dataframe.iloc[:, i])
    plt.plot(pd.Series(results_dataframe.iloc[:, i]).rolling(100, min_periods = 10).mean() )
plt.ylim([0, 100])
plt.rcParams.update({'font.size': 12})
plt.xlabel("Number of episodes", fontsize = 12)
plt.ylabel("Reward", fontsize = 12)
plt.legend(n_values)
plt.show()
 