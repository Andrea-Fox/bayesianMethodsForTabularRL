# %matplotlib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import hsv_to_rgb
import math

import random
from labyrinth import GridWorld


def change_range(values, vmin=0, vmax=1):
    start_zero = values - np.min(values)
    return (start_zero / (np.max(start_zero) + 1e-7)) * (vmax - vmin) + vmin




def egreedy_policy(q_values, state, epsilon=0.1):
    ''' 
    Choose an action based on a epsilon greedy policy.    
    A random action is selected with epsilon probability, else select the best action.    
    '''
    if np.random.random() < epsilon:
        return np.random.choice(4)
    else:
        return np.argmax(q_values[state])


def play(q_values):
    env = GridWorld()
    state = env.reset()
    done = False

    while not done:    
        # Select action
        action = egreedy_policy(q_values, state, 0.0)
        # Do the action
        next_state, reward, done = env.step(action)  

        # Update state and action        
        state = next_state  
        
        env.render(q_values=q_values, action=actions[action], colorize_q=True)





def probabilities_vector(q_values, state, exploration_rate = 0.1):
  
  # prima di tutto troviamo l'azione che massimizza la q-value 
  # (e quindi quella che avrà la probabilità maggiore di essere scelta)

  best_action = np.argmax(q_values[state])

  n_actions = 4 #len(q_values.columns)
   
  # definiamo pra il vettore con le probabilità
  probabilities_vector = np.zeros((n_actions, 1))

  probabilities_vector[:] = exploration_rate/(n_actions -1)

  probabilities_vector[best_action] = 1 - exploration_rate

  return probabilities_vector



def expected_sarsa(env, num_episodes=500, render=True, exploration_rate=0.1,
          learning_rate=0.5, discount_factor=0.9):
    q_values_expected_sarsa = np.zeros((num_states, num_actions))
    ep_rewards = []
    
    for episode_index in range(num_episodes):
        if (episode_index % 5000 == 0):
            print(str(episode_index) + "/" + str(num_episodes))
        state = env.reset()    
        done = False
        reward_sum = 0

        # Choose action        
        action = egreedy_policy(q_values_expected_sarsa, state, exploration_rate)

        while not done:        # we stop when we get to the terminal state
            # Do the action
            next_state, reward, done = env.step(action)
            reward_sum += reward
            
            # Choose next action
            next_action = egreedy_policy(q_values_expected_sarsa, next_state, exploration_rate)

            # Next q value is the value of the next action

            '''
            per calcolare il valore atteso del q_value consideriamo la policy epsilon-greedy,
            che quindi assegna una probabilità pari a 1-eps all'azione  che massimizza il
            q_value e una probabilità uniforme alle altre azioni possibili (che saranno 
            sempre 3 perché supponiamo di poter fare sempre ogni azione, anche se ovviamente 
            alcune ci faranno rimanere nello stesso punto) 
            '''
            vector_probabilities = np.transpose(probabilities_vector(q_values_expected_sarsa, next_state, exploration_rate= exploration_rate ))
            
            # print("vector probabilities = ", vector_probabilities)
            # print("q values = ", q_values_expected_sarsa[next_state])


            expected_q_value = np.sum(q_values_expected_sarsa[next_state]*vector_probabilities)

            td_target = reward + discount_factor * expected_q_value  
            td_error = td_target - q_values_expected_sarsa[state][action]

            # Update q value
            q_values_expected_sarsa[state][action] += learning_rate * td_error

            # Update state and action        
            state = next_state
            action = next_action
            
            if render:
                env.render(q_values, action=actions[action], colorize_q=True)
                
        ep_rewards.append(reward_sum)
        
    return ep_rewards, q_values_expected_sarsa



UP = 0
DOWN = 1
RIGHT = 2
LEFT = 3
actions = ['UP', 'DOWN', 'RIGHT', 'LEFT']

env = GridWorld()


# The number of states in simply the number of "squares" in our grid world, in this case 4 * 12
num_states = 10 * 10
# We have 4 possible actions, up, down, right and left
num_actions = 4

q_values = np.zeros((num_states, num_actions))



df = pd.DataFrame(q_values, columns=[' up ', 'down', 'right', 'left'])
df.index.name = 'States'

expected_sarsa_rewards, q_values = expected_sarsa(env, num_episodes=15000, exploration_rate = 0.1,  discount_factor = 0.95, learning_rate = 0.1, render=False)
play(q_values)

pd.DataFrame.to_csv(pd.DataFrame(expected_sarsa_rewards), sep = ',', path_or_buf="data/returns_expected_sarsa.csv")


