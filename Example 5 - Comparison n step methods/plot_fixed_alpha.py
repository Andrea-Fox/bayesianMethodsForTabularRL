from numpy.core.fromnumeric import argmin
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

dataframe_values = pd.read_csv("data/fixed_alpha_results.csv")

print(dataframe_values)
dataframe_values = dataframe_values.iloc[:, 1:]

dataframe_values.columns = ["k", "n", "alpha", "RMSE"]

print(dataframe_values)

unique_k_values = sorted(list(set(dataframe_values.iloc[:, 0])))
len_unique_k_values = len(unique_k_values)
print("unique k values = ",unique_k_values)
print(len_unique_k_values)

unique_n_values = sorted(list(set(dataframe_values.iloc[:, 1])))
len_unique_n_values = len(unique_n_values)
print(len_unique_n_values)

fig = plt.figure(figsize=(10,6))
for k in range(1,len_unique_k_values-1): 
    plt.plot(dataframe_values.iloc[(k-1)*len_unique_n_values:k*len_unique_n_values, 1], dataframe_values.iloc[(k-1)*len_unique_n_values:k*len_unique_n_values, -1])

plt.rcParams.update({'font.size': 14})
plt.ylabel("RMSE", fontsize = 16)
plt.xlabel("n", fontsize = 16)
plt.legend(["k = " + str(k)  for k in unique_k_values], ncol = 2)
plt.show()


# proviamo ora a cercare il valore minimo per ogni valore
min_list = []

for k in range(1,len_unique_k_values+1):
    min_value = np.argmin(dataframe_values.iloc[(k-1)*len_unique_n_values:k*len_unique_n_values, -1])
    print(unique_k_values[k-1], unique_n_values[min_value])
    # plt.plot(dataframe_values.iloc[(k-1)*len_unique_n_values:k*len_unique_n_values, 1], dataframe_values.iloc[(k-1)*len_unique_n_values:k*len_unique_n_values, -1])

