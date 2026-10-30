# initialize open ai gym and other tools
import gymnasium as gym
import numpy as np
from gymnasium import spaces
from stable_baselines3.common.env_checker import check_env
import matplotlib.pyplot as plt

# note to self: indexing begins at 0

class FlightEnv(gym.Env):
    def __init__(self):
        super(FlightEnv, self).__init__()

        # initialize boeing 737 parameters
        self.mass = 110000.0 # mass of the aircraft [lbs]
        self.fuel = self.np_random.uniform(low=10000, high=46000, size=(1,))[0] # initial fuel amount [lbs]
        self.g = 32.2 # gravitational constant [ft/s^2]
        self.weight = (self.mass + self.fuel) * self.g # weight of the aircraft [lbf]
        self.stall_speed = 128 # stall speed [knots]
        self.thrust = self.np_random.uniform(low=14000, high=29000, size=(1,))[0] # initial thrust [lbf]

        # initialize simulation parameters
        self.dt = 1/360 # time step [hr]
        self.cruise_start = self.np_random.uniform(low=500, high=5000, size=(1,))[0] # total cruise  distance [nautical miles]
        self.cruise_end = 0.0 # cruise end distance [nautical miles]

        # initialize actions and observations
        low_bounds = np.array([-1.0, -1.0]) # min allowable value to alter [thrust, altitude]
        high_bounds = np.array([1.0, 1.0]) # max allowable value to alter [thrust, altitude]
        self.action_space = spaces.Box(low=low_bounds, high=high_bounds, dtype=np.float32) # continuous action space for thrust and altitude adjustments
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(8,), dtype=np.float32)  # weight, wind speed, cruise speed, target cruise speed, altitude, target altitude, remaining fuel, remaining distance to cruise end

    def reset(self, seed=None, options=None):
        super().reset(seed=seed) # reset the environment with the given seed
        self.state = np.zeros(8, dtype=np.float32) # initialize the state array
        
        # point mass model initial state
        self.state[0] = self.weight # weight
        self.state[1] = self.np_random.uniform(low=0, high=20, size=(1,))[0] # wind speed
        self.state[2] = self.np_random.uniform(low=self.stall_speed, high=545, size=(1,))[0] # cruise speed capped between stall speed and maximum speed
        self.state[3] = self.np_random.uniform(low=400, high=500, size=(1,))[0] # target cruise speed based on research
        self.state[4] = self.np_random.uniform(low=30000, high=41000, size=(1,))[0] # altitude
        self.state[5] = self.np_random.uniform(low=33000, high=41000, size=(1,))[0] # target altitude based on research
        self.state[6] = self.fuel # remaining fuel
        self.state[7] = self.cruise_end # remaining distance to cruise end
        
        return self._obs(), {}

    def step(self, action):
        # apply action to the environment
        self.thrust += action[0] # adjust thrust
        self.state[4] += action[1] # adjust altitude

        # update drag and cruise speed
        self.drag = self.state[1]**2 # update drag based on wind speed
        self.state[2] = self.thrust - self.drag # update cruise speed via thrust-drag relationship
        
        # adjust fuel and weight
        self.state[6] -= self.thrust*0.5 # fuel consumption for 737
        self.state[0] = (self.mass-self.thrust*0.5)*self.g # update weight in the state
        self.state[1] = self.np_random.uniform(low=0, high=20, size=(1,))[0] # update wind speed in the state
    
        # calculate reward based on how close the current state is to the target state
        p = 1.2 # penalty factor
        fuel_used = p*(self.thrust*0.5) # amount of fuel used in step
        cruise_speed_diff = p*abs(self.state[2] - self.state[3]) # difference between current and target cruise speed
        altitude_diff = p*abs(self.state[4] - self.state[5]) # difference between current and target altitude
        stall_penalty = 0 # initialize stall penalty
        if self.state[2] < self.stall_speed:
            stall_penalty = p*10.0 # penalty for being below stall speed
        reward = - (cruise_speed_diff + altitude_diff + stall_penalty + fuel_used) # negative reward for being far from target
        
        distance_traveled = self.state[2] * self.dt # distance traveled in the current time step [nautical miles]
        self.state[7] = self.cruise_start - distance_traveled # update remaining distance to cruise end
    
        # initialize goal parameters
        terminated = False # initialize terminated flag
        truncated = False # initialize truncated flag

        if self.cruise_end <= 0:
            terminated = True # mark the episode as terminated if cruise end is reached

        return self._obs(), reward, terminated, False, {"cost": reward}

    def _obs(self):
        return self.state

# check the environment
env = FlightEnv()
check_env(env, warn=True) 
print("No issues found")

obs, info = env.reset()

# plot initialization
plt.figure(figsize=(6, 4))
plt.axis("off")
plt.title("Initial State")


# random steps
for _ in range(5):
  action = env.action_space.sample()  # Random action
  obs, reward, terminated, truncated, info = env.step(action)

  plt.title(f"Action: {action} | Reward: {reward}")
  plt.axis("off")
  plt.show()

  if terminated or truncated:
    obs, info = env.reset()

plt.show()
env.close()