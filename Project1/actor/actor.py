import numpy as np
import random


class Actor:
    def __init__(self, eps, alpha, discount_factor, trace_decay):
        # Set params needed
        self.eps = eps
        self.trace_decay = trace_decay
        self.alpha = alpha
        self.discount_factor = discount_factor
        self.policy = {}
        self.eligibility = {}

        return

    # For linje 2 i algoritmen
    def choose_action(self, env):
        """The agent chooses an action"""

        rand_num = np.random.random()

        # Get the legal actions for the current environment state
        legal_actions = env.legal_actions()

        # If the random number is less than epsilon, choose a random action
        if rand_num < self.eps:
            action = random.choice(legal_actions)

            # If the state-action pair isn't already recorded as a policy, add it with value equals zero
            if (env.state(), action) not in self.policy:
                self.policy[(env.state(), action)] = 0
            return action

        else:
            # Find the action that leads to the highest reward

            # Initialize an action to return if the policy dict is empty
            best_action = legal_actions[0]

            # If the policy dict isn't empty, look up the best action
            if self.policy:
                best_action = max(self.policy, key=self.policy.get)

            return best_action

    # For linje 3, 6c og d i algoritmen
    def update_policy(self, delta, sequence):  # Receive delta from the critic
        """Update eligibility and policy"""

        # Update eligibility for last action: e(s, a) <- 1
        self.eligibility[sequence[-1]] = 1

        # Update policy and eligibility
        for state, action in sequence:
            self.policy[state, action] += self.alpha * delta * self.eligibility[state, action]

            # Question: will the eligibility for the previous state action pairing (s, a) (the one we initialize
            # on line 51) also decrease here?
            self.eligibility[state, action] = self.discount_factor * self.trace_decay * self.eligibility[state, action]
        pass
