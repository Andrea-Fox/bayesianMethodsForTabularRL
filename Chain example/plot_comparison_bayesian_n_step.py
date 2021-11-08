import pandas as pd 
import matplotlib.pyplot as plt


action_selection_types = ["Greedy_selection", "Q-value sampling", "Myopic VPI selection"]
# q_values_updating_types = ["moment_updating"] # ["moment_updating", "mixture_updating"]
# lista_possibili_metodi = [(action_selection_method, q_value_updating_method) for action_selection_method in action_selection_types for q_value_updating_method in q_values_updating_types ]

results_dataframe = pd.read_csv("data/comparison_rewards_bayesian_expected_sarsa_final.csv")

# print(results_dataframe_q)

colors = [ [255/256, 103/256, 0], [86/256, 130/256, 89/256], [139/256, 38/256, 53/256] ] #, [0/256, 78/256, 152/256] ]

fig = plt.figure(figsize=(8, 6))
for i in range(len(action_selection_types)):
    plt.plot(pd.Series(results_dataframe.iloc[:, i]).rolling(25, min_periods = 10).mean() , color = colors[i] )
    
    

for i in range(len(action_selection_types)):
    plt.plot(results_dataframe.iloc[:, i] , "-", color = colors[i] , alpha = 0.3)


plt.ylim([0, 100])
plt.rcParams.update({'font.size': 16})    
plt.xlabel("Number of episodes", fontsize = 16)
plt.ylabel("Reward", fontsize = 16)
plt.title("Bayesian Expected Sarsa", fontsize = 20)
plt.legend(["Greedy selection", "Q-value sampling", "Myopic-VPI selection"] )
plt.show()

