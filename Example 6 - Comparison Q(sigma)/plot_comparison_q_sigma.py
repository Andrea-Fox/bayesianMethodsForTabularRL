import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

dataframe_values_fixed = pd.read_csv(r"C:/Users/Andrea/Dropbox/AndreaFox/codice/random_walk_q_sigma/risultati_numerici/risultato_q_sigma.csv")
dataframe_values_dynamic = pd.read_csv(r"C:/Users/Andrea/Dropbox/AndreaFox/codice/random_walk_q_sigma/risultati_numerici/risultato_q_sigma_dynamic.csv")

# dataframe_values = pd.read_csv("/home/andrea/Desktop/cartella_cluster/risultato_q_sigma.csv")

# dataframe_values_dynamic = pd.read_csv("/home/andrea/Desktop/tesiReinforcmentLearning/codice/random_walk_q_sigma/tabelle/risultato_q_sigma_dynamic.csv")
dataframe_values = pd.concat([dataframe_values_fixed, dataframe_values_dynamic])

# dataframe_values = dataframe_values.iloc[:30, ]



print(dataframe_values)
# unique_k_values = sorted(list(set(dataframe_values.iloc[:, 0])))
# len_unique_k_values = len(unique_k_values)
# print("unique k values = ",unique_k_values)
# print(len_unique_k_values)
# 
# unique_n_values = sorted(list(set(dataframe_values.iloc[:, 1])))
# len_unique_n_values = len(unique_n_values)
# print(len_unique_n_values)

unique_n_episodes_values = sorted(list(set(dataframe_values.iloc[:, 3])))
len_unique_n_episodes_values = len(unique_n_episodes_values)
print(unique_n_episodes_values)
print(len_unique_n_episodes_values)


unique_sigma_values = [0, 0.25, 0.5, 0.75, 1, "dynamic"] # sorted(list(set(dataframe_values.iloc[:, 4])))
len_unique_sigma_values = len(unique_sigma_values)
print(unique_sigma_values)
print("lunghezza = ", len_unique_sigma_values)
# 
important_sigma_values = [0, 1, "dynamic"]
fig = plt.figure(figsize=(10,6))

for k in range(1, len_unique_sigma_values+1):
    print(k, unique_sigma_values[k-1])
    if unique_sigma_values[k-1] in important_sigma_values: 
        print(unique_sigma_values[k-1])
        plt.plot(unique_n_episodes_values, dataframe_values.iloc[(k-1)*len_unique_n_episodes_values:(k)*len_unique_n_episodes_values, -1])
    else:
        plt.plot(unique_n_episodes_values, dataframe_values.iloc[(k-1)*len_unique_n_episodes_values:(k)*len_unique_n_episodes_values, -1], alpha = 0.25)

plt.rcParams.update({'font.size': 15})
plt.ylabel("RMSE", fontsize = 17)
plt.xlabel("Number of episodes", fontsize = 17)
plt.legend([r"$\sigma$ = " + str(k)  for k in unique_sigma_values] , ncol = 3)
plt.show()

