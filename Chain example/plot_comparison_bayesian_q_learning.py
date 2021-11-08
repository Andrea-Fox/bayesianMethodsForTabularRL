import pandas as pd
import matplotlib.pyplot as plt

lista_possibili_metodi = ["Greedy selection", "Q-value sampling", "Myopic VPI selection"]

results_dataframe = pd.read_csv("data/comparison_rewards_bayesian_1_step_q_learning_final.csv")

print(results_dataframe)

colors = [ [255/256, 103/256, 0], [86/256, 130/256, 89/256], [139/256, 38/256, 53/256] ] #, [0/256, 78/256, 152/256] ]

fig = plt.figure(figsize=(8, 6))
for i in range(len(lista_possibili_metodi)):
    # plt.plot(results_dataframe.iloc[:, i])
    plt.plot(pd.Series(results_dataframe.iloc[:, i]).rolling(100, min_periods = 10).mean(),  color = colors[i] )

for i in range(len(lista_possibili_metodi)):
    plt.plot(results_dataframe.iloc[:, i] , "-", color = colors[i] , alpha = 0.3)
plt.ylim([0, 100])
plt.rcParams.update({'font.size': 16})
plt.xlabel("Number of episodes", fontsize = 16)
plt.ylabel("Reward", fontsize = 16)
plt.title("Bayesian Q-learning", fontsize = 20)

plt.legend(lista_possibili_metodi)
plt.show()