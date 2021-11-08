import numpy as np

class Chain():

    def __init__(self, num_states, simulation_index):
        self.num_states = num_states
        self.max_episodes_steps = 20
        self.simulation_index = simulation_index
        return

    def step(self, state, action):
        slip_probability = 0.2
        
        random_number = np.random.random(1)
        if random_number[0] < slip_probability:
            # we do the opposite action
            action = (action+1)%2
        
        if action == 0: # action = 0, go right
            if state == self.num_states -1:
                reward = 10
                next_state = self.num_states-1
            else:
                reward = 0
                next_state = state + 1
        elif action == 1:
            reward = 2
            next_state = 0
        else:
            print(action)
        done = False
        return next_state, reward, done

    def reset(self):
        state = 0
        # state = np.random.choice(5)
        return state