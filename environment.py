import gymnasium as gym
import numpy as np
from gymnasium import spaces

class FlightEnv(gym.Env):
    def __init__(self):
        super(FlightEnv, self).__init__()
        # Define action and observation spaces
        self.action_space = spaces.Discrete(5)  # [+ throttle lg, + throttle sm, hold, - throttle sm, - throttle lg]
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(5,), dtype=np.float32)  # Example: [altitude, speed, fuel, etc.]

    def reset(self, seed=None, options=None):
        # Reset the environment and return the initial observation
        super().reset(seed=seed)
        self.state = np.zeros(5)  # Random initial state