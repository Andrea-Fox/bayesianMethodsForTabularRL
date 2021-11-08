import random
import numpy as np
from numpy.core.fromnumeric import mean
import pandas as pd
import matplotlib.pyplot as plt
import math

import multiprocessing as mp
from markov_reward_process import Markov_Reward_Process


# definiamo il prodotto scalare tra liste
def dot(K, L):
   if len(K) != len(L):
      return 0

   return sum(i[0] * i[1] for i in zip(K, L))

def rmse(predictions, targets):
    return np.sqrt(((predictions - targets) ** 2).mean())

# we need to define the n-step method
def n_step_sarsa(env, num_episodes=500, learning_rate=0.5, discount_factor=1, n = 5, num_states = 50):
    #print("n_step sarsa ", learning_rate, " ", n)
    ep_rewards = []
    value_function = np.ones(num_states+2)*0.5
    value_function[-2: ] = 0

    for _ in range(num_episodes):
        
        future_states = []
        future_rewards = []
        
        t = 0
        tau = t - n +1
        terminal_time = math.inf
        

        state = env.reset()
        # we add S_0 to the list of states
        future_states.append(state)   

        # we add R_0 = 0 to the list of rewards
        future_rewards.append(0)
        done = False
        reward_sum = 0

        while tau != terminal_time -1 :   

            if (t < terminal_time):
                # take action A_t
                next_state, reward, done = env.step()

                # observe and store the next reward as R_{t+1} and the next state as S_{t+1}
                # indeed thay are going to be element t+1 in the respective lists
                future_states.append(next_state)
                reward_sum += reward
                future_rewards.append(reward)    
                
                # If S_{t+1} is terminal, then update terminal_time
                if done:
                    terminal_time = t+1
                    # print("terminal time = ", terminal_time)
                            
            tau = t - n + 1
            if tau >= 0:
                final_index = min(terminal_time, tau + n)
                gamma_list = [discount_factor**x for x in list(range(0, final_index+1-tau-1))]
                final_return = dot(gamma_list, future_rewards[(tau+1):(final_index+1)])
                # print("final_return =", final_return)
                if (tau + n) < terminal_time:
                    final_return  += (discount_factor ** n) * value_function[ future_states[tau+n] ]
                
                # update of q_values table
                value_function[ future_states[tau] ] += learning_rate * (final_return - value_function[ future_states[tau] ] )

            t += 1
            
            
        ep_rewards.append(reward_sum) 
        
    return value_function


def wrapper_parallel_computing(n_episodes= 500, learning_rate = 0.1, discount_factor = 1, n = 1, n_simulations = 50, num_states = 50):
    # print("num states= ", num_states)
    # print("learning rate =  ", learning_rate)
    # print("n = ", n)
    env = Markov_Reward_Process(number_of_states=num_states)
    optimal_values = [i/(num_states+1) for i in range(1, num_states+1)]

    average_values = np.zeros((num_states, 1))
    for simulation_index in range(n_simulations)
        np.random.seed(simulation_index)
        single_simulation_values = n_step_sarsa(env, num_episodes = n_episodes, learning_rate = learning_rate,  discount_factor = discount_factor, n = n, num_states = num_states)
        single_simulation_values = single_simulation_values[0:num_states]

        # fare media tra tutti i valori delle simulazioni 
        average_values = (simulation_index * average_values + single_simulation_values)/(simulation_index +1) 

    # we now compute the RMSE
    mean_squared_error = rmse(average_values, optimal_values)
    print( [num_states, n, learning_rate, mean_squared_error])
    return [num_states, n, learning_rate, mean_squared_error]

# tutte le altre cose per inizializzare
print("Number of processors: ", mp.cpu_count())

# nel nostro caso possiamo fare asyncronous parallel computing

# per sicurezza lasciamo un core libero che non si sa mai con tutti che vanno al 100%
pool = mp.Pool( mp.cpu_count()-1)



n_episodes = 50
n_simulations = 100

# n_values_list = [1, 2, 5, 10, 25, 50, 100, 250]
alpha_values = [0.01] +[i/100 for i in range(2, 19, 2)]+[i/100 for i in range(20, 101, 5)]

num_states = [20] #[5, 20, 50, 100, 1000]
n_values_list = [1, 5, 10, 25, 50] 
# alpha_values = [0.01, 0.05, 0.1, 0.5]


results = []


# optimal_values = [-i/(num_states+1) for i in range(num_states, 0, -1)]

# primo grafico: numero di stati fissato e mostriamo come cambiano il vallore di RMSE a seconda dei valori di n e alpha
lista_possibili_gruppi_di_parametri = [(k, n, alpha) for k in num_states for n in n_values_list for alpha in alpha_values]
print(lista_possibili_gruppi_di_parametri)
# for n, alpha in lista_possibile_coppie_di_parametri:
#     # print(n, alpha)
# alpha = 0.01
results = pool.starmap(wrapper_parallel_computing, [(n_episodes, alpha, 1,  n, n_simulations , k) for k, n, alpha  in lista_possibili_gruppi_di_parametri])  # , callback= collect_result))


pool.close()

dataframe_values = pd.DataFrame(results, columns= ["number of states", "n of steps", "alpha", "RMSE"]  )

dataframe_values.to_csv("data/dataframe_values_20_states.csv", index = False)


