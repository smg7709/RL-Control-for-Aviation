# initialize open ai gym
import gymnasium as gym
import numpy as np
from gymnasium import spaces

class FlightEnv(gym.Env):
    def __init__(self):
        super(FlightEnv, self).__init__()
        # initialize actions and observations
        low_bounds = np.array([-1.0, -0.75]) # min allowable value to alter [cruise speed, altitude]
        high_bounds = np.array([1.0, 0.75]) # max allowable value to alter [cruise speed, altitude]
        self.action_space = spaces.Box(low=low_bounds, high=high_bounds, dtype=np.float32) # continuous action space for cruise speed and altitude adjustments
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(8,), dtype=np.float32)  # weight, wind speed, wind dir, cruise speed, target cruise speed, altitude, target altitude, remaining fuel

    def reset(self, seed=None, options=None):
        # reset the environment and return the initial observation
        super().reset(seed=seed)
        self.state = np.zeros(8)  # random initial state