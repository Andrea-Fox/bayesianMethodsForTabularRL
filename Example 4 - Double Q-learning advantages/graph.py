from scipy.stats import norm
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import random

class Graph:
    
    def __init__(self):
        self.player = None  # indicates the position of the player
          
    def reset(self):
        self.player = 0        
        return self.player

    def step(self, state ,action):
        # Possible actions
        if state == 0:
            if action == 0:
                # go right
                self.player = 2
                done = True
                reward = rewards[state][action]
            else:
                self.player = 1
                done = False
                reward = rewards[state][action]
        else:
            # we start from state 1
            self.player = 3
            done = True
            reward =  float(norm.rvs(size = 1)-0.1)
            # rewards[state][action]
                    
        return self.player, reward, done