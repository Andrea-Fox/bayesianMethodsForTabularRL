

'''
Random walk con k stati: ad ogni stato ho una probabilità pari a 0.5 di andare a destra o a sinistra
Ogni transizione è associata ad una reward di 0, ad eccezione di quella che porta dallo stato più a sinistra a quello terminale lì vicino,
a cui è assegnata una reward pari a -1


OSSERVAZIONI:
1) sampling importance ratio sempre pari ad 1
2) probabilità di ogni azioni sempre pari a 0.5 per ogni azioni
3) non consideriamo action-value function ma state-value function

'''
from os import error
import random
from typing import Optional
import numpy as np
from numpy.core.defchararray import index
from numpy.core.fromnumeric import mean
from numpy.lib.arraysetops import isin
import pandas as pd
import math

import multiprocessing as mp



class Markov_Reward_Process:
    
    def __init__(self, number_of_states = 50):
        self.player = None                              # indicates the position of the player
        self.total_states = number_of_states            # total number of states. Will determine all the other conditions and won't change
          
    def reset(self):
        self.player =  math.floor(self.total_states/2)     
        return self.player

    

    def step(self, action):
        # Possible actions
        
        if action == 0:
            if self.player == 0:
                # we go left in the left-most state: we get to the terminal state
                self.player = self.total_states 
                reward = 0
                done = True
            else:
                self.player -= 1
                reward = 0
                done = False
        else:
            # go right
            if self.player == (self.total_states-1):
                # we go right in the right-most state: we get to the terminal state
                reward = 1
                self.player = self.total_states +1
                done = True
            else:
                self.player += 1
                reward = 0
                done = False                    

        return self.player, reward, done



def select_action():
        random_number = random.uniform(0,1)
        if random_number < 0.5:
            return 0
        else:
            return 1 

# definiamo il prodotto scalare tra liste
def dot(K, L):
   if len(K) != len(L):
      return 0

   return sum(i[0] * i[1] for i in zip(K, L))

def rmse(predictions, targets):
    # print("predictions = ", predictions)
    new_predictions = np.zeros((1, predictions.shape[0]))
    for row in range(predictions.shape[0]):
        new_predictions[0, row] = 0.5 * (predictions[row, 0] + predictions[row, 1])
    # print("new_prediction = ", new_predictions)
    # print("target = ", targets)
    # print(new_predictions - targets)
    return np.sqrt(((new_predictions - targets) ** 2).mean())


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
def q_sigma(env, num_episodes=500, learning_rate=0.5, discount_factor=1, n = 5, num_states = 50, sigma = -1, optimal_values = 0):
    #print("n_step sarsa ", learning_rate, " ", n)

    ep_rewards = []
    value_function = np.ones((num_states+2, 2)) * 0.5  
    value_function[-2: ] = 0

    lista_rmse = []

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

        action = select_action()
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
                
                # If S_{t+1} is terminal, then update terminal_time
                if done:
                    terminal_time = t+1
                    # final_reward = reward
                    # print("terminal time = ", terminal_time)
                    delta = reward - value_function[state, action]
                    future_delta.append(delta)

                else:
                    next_action = select_action()
                    future_actions.append(next_action)
                    expected_value = 0.5 * (value_function[next_state, 0] + value_function[next_state, 1])
                    
                    # if next_state > 1 and next_state < num_states:
                    #     expected_value = 0.5 * (value_function[next_state, 0] + value_function[next_state, 1])
                    # elif next_state == 0:
                    #     expected_value = 0.5 * value_function[next_state, 1]
                    # else:
                    #     expected_value = 0.5 * value_function[next_state, 0]
                    
                    # delta = reward  + value_function[next_state] - value_function[state]
                    delta = reward + discount_factor * ( compute_sigma(sigma, index_episode+1) * value_function[next_state, next_action] + (1-compute_sigma(sigma, index_episode+1)) * expected_value ) - value_function[state, action]
                    future_delta.append(delta)
                    # print("differenza = ", future_delta[t] - delta)
                    # non salviamo il sampling importance ratio, che sarà sempre 1
                          
            tau = t - n + 1
            if tau >= 0:
                E = 1
                # return_G = value_function[ future_states[tau] ]
                return_G = value_function[ future_states[tau] , future_actions[tau]]
                final_index = min(terminal_time, tau + n) # il -1 non lo mettiamo così poi nel range alla riga sotto non dobbiamo aggiungere il +1

                for k in range(tau, final_index):

                    for i in range(tau+1, k+1):
                        E = discount_factor * E * ((1 - compute_sigma(sigma, index_episode)) * 0.5 + compute_sigma(sigma, index_episode) )    # lo 0.5 è la probabilità di fare l'azione
                    # E è sempre 1 in questo caso, per come abbiamo scelto sigma, per come sono le percentuale e perché il discount factor è 1
                    # qui ci sarebbe l'aggiornamento dell'importance sampling ratio, che però è sempre 1
                    return_G += E * future_delta[k]
                
                # value_function[ future_states[tau] ] += learning_rate * (final_return - value_function[ future_states[tau] ]) 
                value_function[ future_states[tau], future_actions[tau] ] += learning_rate * (return_G  - value_function[ future_states[tau], future_actions[tau]] )

            # state = next_state   
            t += 1
        # at the end of each episode we can compute the RMSE at this point
        # print(value_function[0:num_states, :])
        # print(rmse(value_function[0:num_states, :], optimal_values ))
        lista_rmse.append(rmse(value_function[0:num_states, :], optimal_values ))
            
        ep_rewards.append(reward_sum) 
        
    return lista_rmse
    # return value_function


def wrapper_parallel_computing(n_episodes= 500, learning_rate = 0.1, discount_factor = 1, n = 1, n_simulations = 50, num_states = 50, sigma = 0):
    env = Markov_Reward_Process(number_of_states=num_states)
    optimal_values = [i/(num_states+1) for i in range(1, num_states+1)]

    average_values = np.zeros((1, n_episodes))
    
    for simulation_index in range(n_simulations):
        np.random.seed(simulation_index)
        single_simulation_values = q_sigma(env, num_episodes = n_episodes, learning_rate = learning_rate,  discount_factor = discount_factor, n = n, num_states = num_states, sigma = sigma, optimal_values = optimal_values)
        single_simulation_values =  np.array(single_simulation_values)        
        average_values = (simulation_index * average_values + single_simulation_values)/(simulation_index +1) 

    mean_squared_error = average_values.tolist()

    # we now compute the RMSE
    
    risultato = [ (num_states, n, learning_rate, int(n_episode+1), sigma, mean_squared_error[0][n_episode] ) for n_episode in range(n_episodes) ]
    
    print( [num_states, n, learning_rate, n_episodes, sigma, mean_squared_error])
    return risultato

# tutte le altre cose per inizializzare

print("Number of processors: ", mp.cpu_count())

# per sicurezza lasciamo un core libero che non si sa mai con tutti che vanno al 100%
pool = mp.Pool( mp.cpu_count()-1 )




n_episodes_values = [100] #[i for i in range(1, 51, 1)]

n_simulations = 100

n_values_list =  [1, 2] + [i for i in range(5, 51, 5)] + [i for i in range(75, 251, 25)] # 1, 2, 5, 10, ..., 45, 50, 60, 70, ..., 240, 250
alpha_values = 0.4 # [0.05] # general optimal value of the learning rate  
num_states = [5, 10, 20, 50, 100] + [i for i in range(150, 501, 50)] # 5, 10, 20, 50, 100, 150, 200, ..., 450, 500

# sigma_values = ["dynamic"]
sigma_values = [0, 0.25, 0.5, 0.75, 1, "dynamic"]

results = []


# optimal_values = [-i/(num_states+1) for i in range(num_states, 0, -1)]

# primo grafico: numero di stati fissato e mostriamo come cambiano il vallore di RMSE a seconda dei valori di n e alpha
lista_possibili_gruppi_di_parametri = [(sigma, n_episodes)  for sigma in sigma_values for n_episodes in n_episodes_values ]
print(lista_possibili_gruppi_di_parametri)
results = pool.starmap(wrapper_parallel_computing, [(n_episodes, alpha_values, 1,  n_values, n_simulations , n, sigma) for sigma, n_episodes  in lista_possibili_gruppi_di_parametri for n_values in n_values_list for n in num_states])  # , callback= collect_result))


# a questo punto dobbiamo separare la lista, in modo da formare un df vero
# 

# dataframe_values = pd.DataFrame( np.zeros((len(sigma_values) * n_values_list[0], 6)) ) 
# 
pool.close()


final_result = []
for i in range(len(results)):
    final_result = final_result + results[i]

dataframe_values = pd.DataFrame(final_result, columns= ["number of states", "n of steps", "alpha", "n_episodes", "sigma", "RMSE"]  )

print(dataframe_values)

dataframe_values.sort_values(by = ["sigma", "n_episodes"], ascending = [True, True], inplace = True)

print(dataframe_values)
dataframe_values.to_csv("/home/andrea/Desktop/tesiReinforcmentLearning/codice/random_walk_q_sigma/tabelle/risultato_q_sigma_dynamic.csv", index = False)
# dataframe_values.to_csv("risultato_q_sigma.csv", index = False)

