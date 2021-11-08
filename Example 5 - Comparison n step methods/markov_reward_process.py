import random
import numpy as np
import pandas as pd
import math

class Markov_Reward_Process:
    
    def __init__(self, number_of_states = 50):
        self.player = None                              # indicates the position of the player
        self.total_states = number_of_states            # total number of states. Will determine all the other conditions and won't change
          
    def reset(self):
        self.player =  math.floor(self.total_states/2)     
        return self.player

    def step(self):
        # Possible actions
        random_number = random.uniform(0, 1)

        if random_number < 0.5:
            # go left
            if self.player == 0:
                # we go left in the left-most state: we get to the terminal state
                self.player = self.total_states 
                reward = 0
                done = True
            else:
                self.player -= 1
                reward = 0
                done = False
        else:
            # go right
            if self.player == (self.total_states-1):
                # we go right in the right-most state: we get to the terminal state
                reward = 1
                self.player = self.total_states +1
                done = True
            else:
                self.player += 1
                reward = 0
                done = False                    

        return self.player, reward, done