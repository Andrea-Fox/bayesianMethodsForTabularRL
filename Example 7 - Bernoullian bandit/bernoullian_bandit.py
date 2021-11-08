import pandas as pd
import numpy as np
import math
import matplotlib.pyplot as plt



# ambiente
class BernoullianBandit():

    def __init__(self, possible_actions, real_probabilities):
        self.parameters_priori_distribution = np.ones((possible_actions, 2))        # alpha, beta for each action
        self.real_parameters = np.ones((1, possible_actions)) * real_probabilities     # have to be defined in advance
        # the parameters are alpha and beta for all actions. At the beginning, they are all equal to one, as we want to have a 

    def step(self, action):
        # given the action, we know the probability of success, using the real probabilities
        # the return is success (1) or failure (0)

        probability_of_success = self.real_parameters[0, action]
        return np.random.binomial(1, probability_of_success)
    
    def update_parameters_bayesian(self, action, success):
        self.parameters_priori_distribution[action, 0] += success
        self.parameters_priori_distribution[action, 1] += 1 - success


def argmaxrand(a):
    '''
    If a has only one max it is equivalent to argmax, otehrwise it uniformly random selects a maximum
    '''

    indeces = np.where(np.array(a) == np.max(a))[0]
    return np.random.choice(indeces)

# metodi di esplorazione

# epsilon-greedy
def egreedy_policy(q_values, state, epsilon=0.1, first_episode = False):
    ''' 
    Choose an action based on a epsilon greedy policy.    
    A random action is selected with epsilon probability, else select the best action.    
    '''
    if first_episode:
        return np.random.choice(3)
    else:
        if np.random.random() < epsilon:
            # we have to choose an action among the non maximizing one
            try:
                return np.random.choice([i for i in range(3) if q_values[state, i] < np.max(q_values[state, :])])
            except:
                return np.random.choice(3)
        else:
            # print(np.argmax(q_values[state, ]))
            return np.argmax(q_values[state, :])

# boltzmann-exploration
def boltzmann_exploration(q_values, state, episode_index):

    # we choose C_t equal to the maximum between the values
    C = max(abs(q_values[state, 0] - q_values[state, 1]), abs(q_values[state, 0] - q_values[state, 1]), abs(q_values[state, 1] - q_values[state, 2]) )
    C = max(C, 0.1)  # still valid according to the definition
    if C==0:
        C = 1
    try:
        exponentials = [math.exp( ( math.log(episode_index) / C ) *  q_values[state, action]   )   for action in range(3)]
    except:
        print(C)
        print(q_values[state, ])
    sum_exponentials = sum(exponentials)
    if sum_exponentials == 0:
        sum_exponentials = 1
        exponentials = [1/3, 1/3, 1/3]
    probabilites_vector = [exponentials[action]/sum_exponentials  for action in range(3) ]

    # now we need to draw an action according to the probabilty given by probabilites_vector
    action = np.random.choice(np.arange(0, 3), p = probabilites_vector)

    return action

# upper confidence bound
def upper_confidence_bound(q_values,exploration_rate,  episode_index, action_count, possible_actions):
    # at first we have to do choose all the actions once
    if episode_index <= possible_actions:
        action = np.random.choice([a for a in range(possible_actions) if action_count[a] == 0  ])
    else:
        action = np.argmax( [  (q_values[0, a] + exploration_rate * math.sqrt(math.log(episode_index)/action_count[a]  ))   for a in range(possible_actions)  ]    )
    action_count[action] += 1
    return action, action_count

# thompson sampling
def thompson_sampling(env, possible_actions):
    # at first we need to define the values of theta, sampled from the appropriate distributions
    theta_vectors = np.zeros((possible_actions, 1))
    for i in range(possible_actions):
        theta_vectors[i] = np.random.beta(env.parameters_priori_distribution[i, 0], env.parameters_priori_distribution[i, 1])
    
    action = np.argmax(theta_vectors)

    return action

def thompson_sampling_greedy(env, possible_actions):
    theta_vectors = np.zeros((possible_actions, 1))
    for i in range(possible_actions):
        theta_vectors[i] = env.parameters_priori_distribution[i, 0] /(env.parameters_priori_distribution[i, 0] + env.parameters_priori_distribution[i, 1])
    
    
    action = argmaxrand(theta_vectors)

    return action


# algoritmo usato per l'aggiornamento dei q-value

# per il momento q-learning, che tanto va benissimo così

def update_estimates(env, num_episodes=500, exploration_rate=0.1, learning_policy = "epsilon-greedy", possible_actions = 10, naive_update = False, learning_rate = 0.1): 

    Q = np.ones((1, possible_actions))*0.5   # we have only one state: it represents the estimate of the probability of success
    actions_list = []
    action_count = np.zeros((k, 1))
    for episode_index in range(1, num_episodes+1):       

        # action selection  
        if learning_policy == "greedy":
            epsilon = 0
            action = egreedy_policy(Q, 0, epsilon, episode_index == 1)
        elif learning_policy == "constant_epsilon":
            epsilon = exploration_rate
            action = egreedy_policy(Q, 0, epsilon, episode_index == 1)
        elif learning_policy == "epsilon_greedy": 
            epsilon = exploration_rate / episode_index # math.pow(episode_index, 1/5)
            action = egreedy_policy(Q, 0, epsilon, episode_index == 1)

        elif learning_policy == "boltzmann_exploration":
            action = boltzmann_exploration(Q, 0, episode_index)
            
        elif learning_policy == "upper_confidence_bound":
            action, action_count = upper_confidence_bound(Q, exploration_rate, episode_index, action_count, possible_actions)
            
        elif learning_policy == "Thompson_sampling":
            action = thompson_sampling(env, possible_actions)

        elif learning_policy == "Thompson_sampling_greedy":
            action = thompson_sampling_greedy(env, possible_actions)

        # Do the action
        actions_list.append(action)
        
        reward = env.step(action)

        # update of the parameters of the priori distribution

        if naive_update:
            # env.naive_update_parameters(action, reward, learning_rate)
            learning_rate = 1/episode_index
            Q[0, action] += learning_rate * (reward - Q[0, action])
        else:
            env.update_parameters_bayesian(action, reward)
            # update of the estimates of theta (useful when not using Thompson Sampling)
            Q[0, action] = env.parameters_priori_distribution[action, 0] / (env.parameters_priori_distribution[action, 0] + env.parameters_priori_distribution[action, 1])
                
    return Q, actions_list


def handle_actions_list(actions_list_list, n_simulations, n_episodes, possible_actions):
    probabilities = np.zeros((n_episodes, possible_actions))
    # probabilities[i, j] = probability of doing action j at time_step i
    for episode_index in range(n_episodes):
        action_counter = np.zeros((1, possible_actions))
        for simulation_index in range(n_simulations):
            action_counter[0, actions_list_list[simulation_index][episode_index]] += 1  
            # increase the counter corresponding to that action at time episode_index
        probabilities[episode_index, ] = action_counter/n_simulations

    return probabilities



# main

k = 3    # number of possible actions

n_simulations = 10000
n_episodes = 1000
actions_list_list = []
real_probabilities = [0.6, 0.75, 0.9]
exploration_rate = 0.1
# learning_policy= "epsilon_greedy"

# learning_policies = ["epsilon_greedy"] #, "Thompson_sampling"]
learning_policies = ["greedy", "constant_epsilon", "epsilon_greedy", "boltzmann_exploration"] #, "upper_confidence_bound", "Thompson_sampling", "Thompson_sampling_greedy"]
for learning_policy in learning_policies:
    actions_list_list = []
    print(learning_policy)
    
    for i in range(n_simulations):
        if (i % 1000 == 0):
                print(str(i) + "/" + str(n_simulations))  
        np.random.seed(i)
        env = BernoullianBandit(k, real_probabilities)
        Q, actions_list = update_estimates(env, num_episodes = n_episodes, exploration_rate= exploration_rate, learning_policy=learning_policy , possible_actions=k, naive_update=True, learning_rate=0.1)
        actions_list_list.append(actions_list)


    # From actions_list, given that n_simulations simulations have been done, it is possible to compute an estimate of the probability of being chosen at a given time step  
    # After all the simulations we should have n_simulations lists of n_epsiodes elements: we need to create a list with all the probabilities for each timestep
    probabilities_list = handle_actions_list(actions_list_list, n_simulations, n_episodes, possible_actions = k)


    probabilities_dataframe = pd.DataFrame(probabilities_list)
    probabilities_dataframe.columns = real_probabilities

    save_data = True
    if save_data:
        if learning_policy == "greedy":
            probabilities_dataframe.to_csv("data/results_with_greedy_naive_update.csv", index=False)
        elif learning_policy == "constant_epsilon": 
            probabilities_dataframe.to_csv("data/results_with_constant_epsilon_naive_update.csv", index=False)
        elif learning_policy == "epsilon_greedy": 
            probabilities_dataframe.to_csv("data/results_with_epsilon_greedy_naive_update.csv", index=False)
        elif learning_policy == "boltzmann_exploration":
            probabilities_dataframe.to_csv("data/results_with_boltzmann_exploration_naive_update.csv", index=False)
        elif learning_policy == "upper_confidence_bound":
            probabilities_dataframe.to_csv("data/results_with_upper_confidence_bound.csv", index=False)
        elif learning_policy == "Thompson_sampling":
            probabilities_dataframe.to_csv("data/results_with_thompson_sampling.csv", index=False)
        elif learning_policy == "Thompson_sampling_greedy":
            probabilities_dataframe.to_csv("data/results_with_thompson_sampling_greedy.csv", index=False)

    plot = True
    if plot:
        fig = plt.figure(figsize=(8,6))
        plt.plot(probabilities_dataframe.iloc[:, 0])
        plt.plot(probabilities_dataframe.iloc[:, 1])
        plt.plot(probabilities_dataframe.iloc[:, 2])
        plt.ylim([0, 1])  
        plt.rcParams.update({'font.size': 12})
        plt.xlabel("Number of episodes", fontsize = 12)
        plt.ylabel("Probability of choosing the action", fontsize = 12)
        plt.title(learning_policy)
        plt.legend([r"$\theta_0 = $"+ str(real_probabilities[0]) , r"$\theta_1 = $" + str(real_probabilities[1]), r"$\theta_2 = $" + str(real_probabilities[2])])
        plt.show()


