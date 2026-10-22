# initialize open ai gym
import gymnasium as gym
import numpy as np
from gymnasium import spaces

# note to self: indexing begins at 0

class FlightEnv(gym.Env):
    def __init__(self):
        super(FlightEnv, self).__init__()
        # initialize actions and observations
        low_bounds = np.array([-1.0, -0.75]) # min allowable value to alter [cruise speed, altitude]
        high_bounds = np.array([1.0, 0.75]) # max allowable value to alter [cruise speed, altitude]
        self.action_space = spaces.Box(low=low_bounds, high=high_bounds, dtype=np.float32) # continuous action space for cruise speed and altitude adjustments
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(8,), dtype=np.float32)  # weight, wind speed, wind dir, cruise speed, target cruise speed, altitude, target altitude, remaining fuel
        self.stall_speed = 60.0 # stall speed in knots

    def reset(self, seed=None, options=None):
        super().reset(seed=seed) # reset the environment with the given seed
        self.state = self.np_random.uniform(low=-1, high=1, size=(8,)) # initial state: weight, wind speed, wind dir, cruise speed, target cruise speed, altitude, target altitude, remaining fuel
        return self._obs(), {}

    def step(self, action):
        # apply action to the environment
        self.state[3] += action[0] # adjust cruise speed
        self.state[5] += action[1] # adjust altitude
    
        # calculate reward based on how close the current state is to the target state
        cruise_speed_diff = abs(self.state[3] - self.state[4]) # difference between current and target cruise speed
        altitude_diff = abs(self.state[5] - self.state[6]) # difference between current and target altitude
        # add stall speed reward
        reward = - (cruise_speed_diff + altitude_diff) # negative reward for being far from target

        # check if the episode is done (e.g., if fuel is depleted)
        done = self.state[7] <= 0 # remaining fuel is zero or less
        return self._obs(), reward, done, False, {}