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



def q_learning(env, num_episodes=500, render=True, exploration_rate=0.1,
               learning_rate=0.5, discount_factor=0.9): 

    q_values = np.zeros((num_states, num_actions))
    ep_rewards = []
    
    for episode_index in range(num_episodes):
        if (episode_index % 5000 == 0):
            print(str(episode_index) + "/" + str(num_episodes))
        state = env.reset()    
        done = False
        reward_sum = 0

        while not done:            
            # Choose action        
            action = egreedy_policy(q_values, state, exploration_rate)
            
            # Do the action
            next_state, reward, done = env.step(action)
            reward_sum += reward

            # Update q_values       
            td_target = reward + discount_factor * np.max(q_values[next_state])   # target found using q-learning
            td_error = td_target - q_values[state][action]
            q_values[state][action] += learning_rate * td_error
            
            # Update state
            state = next_state

            if render:
                env.render(q_values, action=actions[action], colorize_q=True)
        ep_rewards.append(reward_sum)
    
    return ep_rewards, q_values





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

q_learning_rewards, q_values = q_learning(env, num_episodes=100000, exploration_rate = 0.1,  discount_factor = 0.95, learning_rate = 0.1, render=False)
play(q_values)

pd.DataFrame.to_csv(pd.DataFrame(q_learning_rewards), sep = ',', path_or_buf="data/returns_q_learning.csv")
