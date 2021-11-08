import pandas as pd
import matplotlib.pyplot as plt



n_episodes = 1000 #[i for i in range(1, 51, 1)]

n_simulations = 10

n_values = [1, 2, 5, 10, 20]   # [1, 2] + [i for i in range(5, 51, 5)] + [i for i in range(75, 251, 25)] # 1, 2, 5, 10, ..., 45, 50, 60, 70, ..., 240, 250
alpha_values = 0.1 # [0.05] # general optimal value of the learning rate  
num_states = 5 
learning_policies = ["greedy", "constant-epsilon" "epsilon_greedy", "boltzmann_exploration"]
exploration_rates = [0, 0.1, 1.5]
learning_policies_with_parameters = list(zip(learning_policies, exploration_rates))
print(learning_policies_with_parameters)
sigma_values = [0, 0.25, 0.5, 0.75, 1] #, "dynamic"]


# lista_possibili_gruppi_di_parametri = [(0, 1000, "epsilon_greedy", 0.1, 10), (0.75, 1000, "greedy", 0, 5), (1, 1000, "greedy", 0, 10), (1, 1000, "epsilon_greedy", 0.1, 10), (1, 1000, "boltzmann_exploration", 1.5, 10)]
# print(lista_possibili_gruppi_di_parametri)


# primo grafico: numero di stati fissato e mostriamo come cambiano il vallore di RMSE a seconda dei valori di n e alpha
lista_possibili_gruppi_di_parametri = [(0.5, n_episodes, "constant-epsilon", 0.1, 1), (0.75, n_episodes, "greedy", 0, 20), (1, n_episodes, "epsilon_greedy", .1, 5), (0.75, n_episodes, "constant-epsilon", 0.1, 5), (1, n_episodes, "boltzmann_exploration", 1.5, 10)]


dataframe_values = pd.read_csv("data/chain_q_sigma_optimal_values.csv")
lista_legend = []
print(dataframe_values)
colors = [ [255/256, 103/256, 0], [86/256, 130/256, 89/256], [139/256, 38/256, 53/256] ]
fig = plt.figure(figsize=(8, 6))
index_color = 0
for i in range(len(lista_possibili_gruppi_di_parametri)):
    plt.plot(pd.Series(dataframe_values.iloc[:, i]).rolling(50, min_periods = 10).mean() ) #, color = colors[index_color]   )
    # plt.plot(dataframe_values.iloc[:, i], alpha = 0.25 )# , color = colors[index_color])
    
    # index_color += 1

plt.ylim([0, 100])
plt.rcParams.update({'font.size': 12})    
plt.xlabel("Number of episodes", fontsize = 16)
plt.ylabel("Reward", fontsize = 16)
plt.title(r"Q($\sigma$)", fontsize = 20)
# nomi_per_legenda = [r"$\epsilon$-greedy", "Greedy", "Greedy", r"$\epsilon$-greedy", "Boltzmann exploration"]
plt.legend([r"Constant-$\epsilon$-greedy, $\sigma = 0.5$, $n = 1$", r"Greedy exploration, $\sigma = 0.75$, $n = 20$", r"$\epsilon$-greedy, $\sigma = 1$, $n = 20$", r"Constant-$\epsilon$-greedy, $\sigma = 0.75$, $n = 5$", r"Boltzmann exploration, $\sigma = 1$, $n = 10$"]) #[ r"$\sigma$ = " + str(lista_possibili_gruppi_di_parametri[i][0])  +", " + nomi_per_legenda[i]  for i in lista_legend]) #, ncol = 3)
plt.show()

