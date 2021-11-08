from scipy.stats import norm
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import random

from graph import Graph

def egreedy_policy(q_values, state, action, epsilon = 0.1):
    ''' 
    Choose an action based on a epsilon greedy policy.    
    A random action is selected with epsilon probability, else select the best action.    
    We need to separate the action selectd in state 0/A and the action selectd in 1/B
    '''
    if (state == 0):
        if np.min(q_values[state]) == np.max(q_values[state]):
            action = np.random.choice(2)
        else:
            if np.random.random() < epsilon:
                action =  np.random.choice(2)
            elif np.max(q_values[state]) == np.min(q_values[state]):
                action =  np.random.choice(2)
            else:
                action =  np.argmax(q_values[state])
        return action

    elif(state == 1):
        #return np.random.choice(number_of_actions_from_1)
        if np.random.random() < epsilon:
            action =  np.random.choice(number_of_actions_from_1)
        else:
            action =  np.argmax(q_values[state])
        return action
        # return np.argmax(q_values[state][action])



def double_q_learning(env, num_episodes=500, render=True, exploration_rate=0.1,
          learning_rate=0.5, discount_factor=0.9):
    q_1_values_double_learning = [np.zeros(2), np.zeros(number_of_actions_from_1), 0, 0]
    q_2_values_double_learning = [np.zeros(2), np.zeros(number_of_actions_from_1), 0, 0]
    left_actions_counter = 0
    left_actions_percentage_list = []
    
    ep_rewards = []
    aggiornamento_q_1= 0
    n_aggiornamenti = 0

    for episode_index in range(1, num_episodes+1):
        if (episode_index % 5000 == 0):
            print(str(episode_index) + "/" + str(num_episodes))
        state = env.reset()    
        done = False
        reward_sum = 0

        # Choose action        

        # summed_q_values = q_1_values_double_learning + q_2_values_double_learning 
        # action = egreedy_policy(summed_q_values, state, exploration_rate)

        while not done:            
            # Choose action     
            summed_q_values = [ [q_1_values_double_learning[0]/2 + q_2_values_double_learning[0]/2 ],  [q_1_values_double_learning[1]/2 + q_2_values_double_learning[1]/2 ], 0, 0]
            action = egreedy_policy(summed_q_values, state, exploration_rate)

            if (state == 0):
                if action == 1:
                    left_actions_counter += 1
                left_actions_percentage_list.append(left_actions_counter/episode_index)

            
            # Do the action and observe the reward and the following state
            next_state, reward, done = env.step(state, action)
            reward_sum += reward

            random_value = random.uniform(0, 1)

            # Update q_values
            
            # if random_value < 0.5 we update q_1, else we update q_2

            if (random_value < 0.5):
                aggiornamento_q_1 += 1
                if (next_state >= 2 ):
                    td_target = reward
                else:
                    td_target = reward + discount_factor * q_2_values_double_learning[next_state][ np.argmax(q_1_values_double_learning[next_state]) ]   # target found using q-learning
                td_error = td_target - q_1_values_double_learning[state][action]
                q_1_values_double_learning[state][action] += learning_rate * td_error
            else: 
                if (next_state >= 2 ):
                    td_target = reward
                else:
                    td_target = reward + discount_factor * q_1_values_double_learning[next_state][ np.argmax(q_2_values_double_learning[next_state]) ]   # target found using q-learning
                td_error = td_target - q_2_values_double_learning[state][action]
                q_2_values_double_learning[state][action] += learning_rate * td_error
            n_aggiornamenti += 1
             

            
            # Update state
            state = next_state

        ep_rewards.append(reward_sum)
        
    # print("percentuale aggiornamento q_1 = ", aggiornamento_q_1/n_aggiornamenti)
    return ep_rewards, summed_q_values, left_actions_percentage_list

env = Graph()
# we find the 20 random returns for this attempt
number_of_actions_from_1 = 3

random_returns = norm.rvs(size = number_of_actions_from_1)-0.1

print("media = ",np.mean(random_returns))
print(random_returns)

print(np.zeros(number_of_actions_from_1))

# q_values = [list(np.zeros(2)), list(random_returns), 0, 0]

n_episodes = 1000
n_simulations = 5000
average_values = np.zeros(n_episodes)

for simulation_index in range(n_simulations):
    np.random.seed(simulation_index)
    rewards = [list(np.zeros(2)), list(random_returns), 0, 0]

    if (simulation_index % 50 == 0):
        print(str(simulation_index) + "/" + str(n_simulations))
    _, summed_q_values, left_action_percentage = double_q_learning(env, num_episodes=n_episodes, exploration_rate = 0.1,  discount_factor = 1, learning_rate = 0.1)
    # fare media tra tutti i valori delle simulazioni 
    # 
    # print(left_action_percentage)
    # print(summed_q_values)
    average_values = (simulation_index * average_values + left_action_percentage)/(simulation_index +1)


pd.DataFrame.to_csv(pd.DataFrame(average_values), "data/result_double_q_learning.csv")

plt.plot(average_values)
plt.show()