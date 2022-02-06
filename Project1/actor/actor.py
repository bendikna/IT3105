import numpy as np
import random


class Actor:
    def __init__(self, eps, alpha, gamma, l):
        # Set params needed
        self.eps = eps
        self.alpha = alpha
        self.gamma = gamma
        return

# For linje 2 i algoritmen
    def choose_action(self, env):
        rand_num = np.random.random()
        possible_actions = env.possible_actions()
        if rand_num < self.eps:
            action = random.choice(possible_actions)
        else:
            # Finne en måte å velge beste trekk
            return

# For linje 3, 6c og d i algoritmen
    def update_policy(self, delta):  # Delta finner vi fra Value
        # Sjekke om vi har vært i state før, legge til state om False

        # Oppdatere policy
        # policy[s, a] += self.alpha*delta*eligibility
        # eligibility[s, a] = gamma*self.l*eligibility[s, a]
        pass
