'''
proviamo a parallelizzare i passaggi, in modo da ridurre il tempo impiegato
'''

'''
Random walk con k stati: ad ogni stato ho una probabilità pari a 0.5 di andare a destra o a sinistra
Ogni transizione è associata ad una reward di 0, ad eccezione di quella che porta dallo stato più a sinistra a quello terminale lì vicino,
a cui è assegnata una reward pari a -1

Cose da fare:
- prima fare un'analisi seria per un certo valore di k (esempio k = 50) e mostrare quale è il valore di n che porta alla minore RMSE per diversi valori di alfa (che è quello che fa il libro)
- cambiare valore di k e mostrare come ad ogni valore di k corrisponde un n diverso (a questo punto considerare solo quello che è l'alpha ottimale e il valore di n ottimale)
- capire se c'è una relazione tra k e n

'''
import random
import numpy as np
import pandas as pd
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
    for simulation_index in range(n_simulations):
        # print(simulation_index)
        # value_function = np.ones(num_states+2)*0.5
        # value_function[-2: ] = 0
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
pool = mp.Pool( mp.cpu_count())



n_episodes = 50
n_simulations = 100

n_values_list = [3]
# n_values_list = [1, 2] + [i for i in range(5, 51, 5)] + [i for i in range(75, 251, 25)] # 1, 2, 5, 10, ..., 45, 50, 60, 70, ..., 240, 250
alpha_values = [0.4] # general optimal value of the learning rate
num_states = [5, 10, 20, 50, 100] + [i for i in range(150, 501, 50)] # 5, 10, 20, 50, 100, 150, 200, ..., 450, 500
# num_states = [99]

results = []


# optimal_values = [-i/(num_states+1) for i in range(num_states, 0, -1)]

# primo grafico: numero di stati fissato e mostriamo come cambiano il vallore di RMSE a seconda dei valori di n e alpha
lista_possibili_gruppi_di_parametri = [(k, n, alpha) for k in num_states  for n in n_values_list  for alpha in alpha_values]

results = pool.starmap(wrapper_parallel_computing, [(n_episodes, alpha, 1,  n, n_simulations , k) for k, n, alpha  in lista_possibili_gruppi_di_parametri])  # , callback= collect_result))

# 
# 
print("aaa")
print(results)
pool.close()

dataframe_values = pd.DataFrame(results, columns= ["number of states", "n of steps", "alpha", "RMSE"]  )

print(dataframe_values)
dataframe_values.to_csv("data/fixed_alpha_results.csv", index = False)

