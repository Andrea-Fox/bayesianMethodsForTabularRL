''' 
Due tipi di grafici: in un primo si mostra come, all'aumentare del numero di episodi, tutti e tre i metodi si avvicinano 
ai valori ottimali della value function

In un secondo grafico si mostrano gli effettivi calcoli per la error function, con l'empirical RMS error (di cui va trovata anche la formula)
Capire se conviene fare un unico grafico come fatto dal libro oppure più grafici (del tipo che si provano diversi valori di alpha e per ognuno 
di quelli si fa un confronto tra i due metodi)
Un'alternativa è che si trovano per tutti i metodi i valori ottimali di alpha (anche graficamente) e poi si fa un quarto grafico in cui si 
mettono a confronto gli errori minimi di tutti e due i metodi) 

N.B. qua stiamo considerando solo il metodo di TD più semplice, ovvero Sarsa. Usando metodi più sofisticati, come per esempio Q-learning 
oppure Expected Sarsa, si potrebbero ottenere risultati ancora migliori (e questo magari a fine capitolo potrebbe essere mostrato, in un altro esempio)
'''
import numpy as np
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def rmse(predictions, targets):
    # print( (predictions - targets)**2)
    # print( ((predictions - targets) ** 2).mean() )
    # print( np.sqrt(((predictions - targets) ** 2).mean()) )
    # print("-------------------------------------")
    return np.sqrt(((predictions - targets) ** 2).mean())



td_0_1 = pd.read_csv("data/temporal_difference_alpha_0-1.csv")
td_0_01 = pd.read_csv("data/temporal_difference_alpha_0-01.csv")
td_0_03 = pd.read_csv("data/temporal_difference_alpha_0-03.csv")
td_0_05 = pd.read_csv("data/temporal_difference_alpha_0-05.csv")

mc_0_1 = pd.read_csv("data/monte_carlo_alpha_0-1.csv")
mc_0_01 = pd.read_csv("data/monte_carlo_alpha_0-01.csv")
mc_0_03 = pd.read_csv("data/monte_carlo_alpha_0-03.csv")
mc_0_05 = pd.read_csv("data/monte_carlo_alpha_0-05.csv")





td_0_1 = td_0_1.to_numpy()
td_0_01 = td_0_01.to_numpy()
td_0_03 = td_0_03.to_numpy()
td_0_05 = td_0_05.to_numpy()


mc_0_1 = mc_0_1.to_numpy()
mc_0_01 = mc_0_01.to_numpy()
mc_0_03 = mc_0_03.to_numpy()
mc_0_05 = mc_0_05.to_numpy()



nrow = td_0_1.shape[0]
rmse_table = pd.DataFrame(np.zeros((nrow, 8)))
rmse_table.columns = ["temporal_difference_0_1", "temporal_difference_0_1", "temporal_difference_0_03", "temporal_difference_0_05",  "monte_carlo_0_1", "monte_carlo_0_01", "monte_carlo_0_03", "monte_carlo_0_05"]
print(rmse_table)


real_values = [1/6, 2/6, 3/6, 4/6, 5/6]

for i in range(nrow):
    rmse_table.iloc[i, 0] = rmse(td_0_1[i, 2:], real_values)
    rmse_table.iloc[i, 1] = rmse(td_0_01[i, 2:], real_values)
    rmse_table.iloc[i, 2] = rmse(td_0_03[i, 2:], real_values)
    rmse_table.iloc[i, 3] = rmse(td_0_05[i, 2:], real_values)

    rmse_table.iloc[i, 4] = rmse(mc_0_1[i, 2:], real_values)
    rmse_table.iloc[i, 5] = rmse(mc_0_01[i, 2:], real_values)
    rmse_table.iloc[i, 6] = rmse(mc_0_03[i, 2:], real_values)
    rmse_table.iloc[i, 7] = rmse(mc_0_05[i, 2:], real_values)
    

# print(rmse_table)
color_monte_carlo = (0/256, 105/256, 146/256)
color_sarsa =  (158/256, 42/256, 43/256)


plt.plot(rmse_table.iloc[:, 1], '--',color = color_sarsa)
plt.plot(rmse_table.iloc[:, 2], '-.',color = color_sarsa)
plt.plot(rmse_table.iloc[:, 3], color = color_sarsa)
plt.plot(rmse_table.iloc[:, 0], linewidth = 3, color = color_sarsa)

plt.plot(rmse_table.iloc[:, 5], '--', color = color_monte_carlo)
plt.plot(rmse_table.iloc[:, 6], '-.', color = color_monte_carlo)
plt.plot(rmse_table.iloc[:, 7], color = color_monte_carlo)
plt.plot(rmse_table.iloc[:, 4], linewidth = 3, color = color_monte_carlo)

plt.rcParams.update({'font.size': 15})
plt.xlabel("Number of episodes", fontsize = 23)
plt.ylabel("RMSE", fontsize = 23)

plt.legend([ r"TD, $\alpha = 0.01$",r"TD, $\alpha = 0.03$",r"TD, $\alpha = 0.05$",r"TD, $\alpha = 0.1$", r"MC, $\alpha = 0.01$", r"MC, $\alpha = 0.03$", r"MC, $\alpha = 0.05$",  r"MC, $\alpha = 0.1$"])
plt.show()


