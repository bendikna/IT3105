import numpy as np


class TableCritic:
    def __init__(self, lr, trace_decay, discount_factor, alpha):
        # Set parameters
        self.lr = lr
        self.trace_decay = trace_decay
        self.discount_factor = discount_factor
        self.alpha = alpha
        self.eligibility = {}
        self.state_values = {}
        self.delta = 0

    # Getting delta error
    def get_error_delta(self, reward, state, state_new):
        if state not in self.state_values:  # Check if current state in values
            self.state_values[state] = np.random.uniform(0, 0.1)  # Init state with small values
        if state_new not in self.state_values:  # Check if current state in values
            self.state_values[state_new] = np.random.uniform(0, 0.1)  # Init state with small values

        self.delta = reward + self.discount_factor * self.state_values[state] - self.state_values[state_new]
        return self.delta

    def update_state_values(self, sequence):
        self.eligibility[sequence[-1]] = 1  # Set elig to 1
        for state, value in sequence:  # Update eligibility and state_values
            self.state_values[state] += self.alpha * self.delta * self.eligibility[state]

            self.eligibility[state] = self.discount_factor * self.trace_decay * self.eligibility[state]
