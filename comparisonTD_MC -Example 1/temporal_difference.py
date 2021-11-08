import numpy as np
import pandas as pd
import numpy as np
import random
import matplotlib.pyplot as plt


'''
Obiettivo: trovare la value function per ogni nodo
La posizione iniziale sarà sempre il nodo C. A partire da ogni nodo si può andare sia a destra che a sinistra, con probabilità 0.5 di fare ciascuna mossa
Gli unici stati terminali sono quelli agli estremi. Le reward sono tutte 0, ad eccezione di quella che porta dallo stato E a quello terminale vicino; 
in quel caso la reward è pari ad 1.

Per comodità (e per snellire il codice) i nodi interni saranno identificati con numeri da 0 a 4 (0 = A, 1 = B, 2 = C, 3 = D, 4 = E), mentre agli stati
terminali vengono assegnati gli indici 5 e 6
'''

class Markov_Reward_Process:
    
    def __init__(self):
        self.player = None  # indicates the position of the player
          
    def reset(self):
        self.player = 2        
        return self.player

    def step(self):
        # Possible actions
        random_number = random.uniform(0, 1)

        if random_number < 0.5:
            # go left
            if self.player == 0:
                self.player = 5
                reward = 0  
                done = True
            else:
                self.player -= 1
                reward = 0
                done = False
        else:
            # go right
            if self.player == 4:
                reward = 1
                self.player = 6
                done = True
            else:
                self.player += 1
                reward = 0
                done = False                    

        return self.player, reward, done


def temporal_difference_prediction(env, num_episodes=500, learning_rate=0.1, discount_factor=1):
        
    for _ in range(num_episodes):
        state = env.reset()    
        done = False
        reward_sum = 0

        # Choose action        
        while not done:        # we stop when we get to the terminal state
            # Do the action
            next_state, reward, done = env.step()
            reward_sum += reward
            
            # Next q value is the value of the next action
            td_target = reward + discount_factor * values_sarsa[next_state]
            td_error = td_target - values_sarsa[state]

            # Update q value
            values_sarsa[state] += learning_rate * td_error

            # Update state and action        
            state = next_state
            
        
    return values_sarsa



env = Markov_Reward_Process()
# we find the 20 random returns for this attempt

num_states = 7

values_sarsa = np.ones(num_states)*0.5
values_sarsa[-2: ] = 0
print(values_sarsa)

n_episodes = 100
n_simulations = 100
alpha = 0.1

dataframe_values = pd.DataFrame(np.zeros((n_episodes, 6)))
dataframe_values.columns = ["episodes_considered", "V(0)", "V(1)", "V(2)", "V(3)", "V(4)"]

for number_episodes_index in range(1, n_episodes+1):
    average_values = np.zeros(5)
    if number_episodes_index % 50 == 0:
            print(str(number_episodes_index) + "/" + str(n_episodes))
    
    for simulation_index in range(n_simulations):
        np.random.seed(simulation_index)
        values_sarsa = np.ones(num_states)*0.5
        values_sarsa[-2: ] = 0
        
        single_simulation_values = temporal_difference_prediction(env, num_episodes = number_episodes_index, learning_rate = alpha,  discount_factor = 1)
        single_simulation_values = single_simulation_values[0:5]
        # fare media tra tutti i valori delle simulazioni 
        average_values = (simulation_index * average_values + single_simulation_values)/(simulation_index +1) 

    # assegnare alla giusta posizione nel dataframe i valori medi ottenuti
    dataframe_values.iloc[number_episodes_index-1, 0] = number_episodes_index
    dataframe_values.iloc[number_episodes_index-1, 1:] = average_values

dataframe_values.to_csv("data/temporal_difference_alpha_0-1.csv")

print(dataframe_values)


plt.plot(dataframe_values.iloc[0, 1:])
plt.plot(dataframe_values.iloc[9, 1:])
plt.plot(dataframe_values.iloc[49, 1:])
plt.plot(dataframe_values.iloc[99, 1:])
plt.scatter(x = [0, 1, 2, 3, 4], y = [1/6, 2/6, 3/6, 4/6, 5/6])
plt.legend(["1 episode", "10 episode", "50 episodes", "100 episodes"]) 
plt.show()


