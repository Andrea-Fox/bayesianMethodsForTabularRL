from scipy.stats import norm
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


from graph import Graph


def egreedy_policy(q_values, state, epsilon=0.1):
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
        r# eturn np.argmax(q_values[state][action])


    



def q_learning(env, num_episodes=500, render=True, exploration_rate=0.1,
               learning_rate=0.5, discount_factor=0.9): 
    ep_rewards = []

    left_actions_counter = 0
    left_actions_percentage_list = []

    
    for episode_index in range(1, num_episodes+1):
        if (episode_index % 5000 == 0):
            print(str(episode_index) + "/" + str(num_episodes))
        state = env.reset()    
        done = False
        reward_sum = 0

        while not done:            
            # Choose action        
            action = egreedy_policy(q_values, state, exploration_rate)
            
            if (state == 0):
                if action == 1:
                    left_actions_counter += 1
                left_actions_percentage_list.append(left_actions_counter/episode_index)

            

            # Do the action
            next_state, reward, done = env.step(state, action)
            reward_sum += reward

            # Update q_values       
            td_target = reward + discount_factor * np.max(q_values[next_state])   # target found using q-learning
            td_error = td_target - q_values[state][action]
            q_values[state][action] += learning_rate * td_error
            
            # Update state
            state = next_state

        ep_rewards.append(reward_sum)
    
    return ep_rewards, q_values, left_actions_percentage_list

env = Graph()
# we find the 20 random returns for this attempt
number_of_actions_from_1 = 2

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
    q_values = [list(np.zeros(2)), list(np.zeros(number_of_actions_from_1)), 0, 0]
    rewards = [list(np.zeros(2)), list(random_returns), 0, 0]

    if (simulation_index % 50 == 0):
        print(str(simulation_index) + "/" + str(n_simulations))
    q_learning_rewards, q_values, left_action_percentage = q_learning(env, num_episodes=n_episodes, exploration_rate = 0.1,  discount_factor = 1, learning_rate = 0.1)
    # fare media tra tutti i valori delle simulazioni 
    average_values = (simulation_index * average_values + left_action_percentage)/(simulation_index +1)


pd.DataFrame.to_csv(pd.DataFrame(average_values), "data/result_q_learning.csv")

# print(left_action_percentage)


plt.plot(average_values)
plt.show()