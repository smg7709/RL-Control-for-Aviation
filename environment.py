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
        self.empty = 110000 # weight of the aircraft [lbs]
        self.fuel = 40000 # weight of fuel [lbs]
        self.weight = self.empty + self.fuel # total aircraft weight [lbs]
        self.stall_speed = 128 # stall speed [knots]
        self.c_drag = 0.049 # coefficient of drag [-]
        self.thrust = 5000 # initial thrust [lbf]

        # initialize simulation parameters
        self.dt = 1/360 # time step [hr]
        self.dts = self.dt*3600 # time step [s]
        self.cruise_start = 0
        self.cruise_end = self.np_random.uniform(low=500, high=5000, size=(1,))[0] # total cruise distance [nautical miles]
        
        # initialize actions and observations
        low_bounds = np.array([-1, -1]) # min allowable value to alter [thrust [lbf], altitude [ft]]
        high_bounds = np.array([1, 1]) # max allowable value to alter [thrust [lbf], altitude [ft]]
        self.action_space = spaces.Box(low=low_bounds, high=high_bounds, dtype=np.float32) # continuous action space for thrust and altitude adjustments
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(9,), dtype=np.float32)  # weight, wind speed, cruise speed, target cruise speed, altitude, target altitude, remaining fuel, remaining distance to cruise end, thrust

    def reset(self, seed=None, options=None):
        super().reset(seed=seed) # reset the environment with the given seed
        self.state = np.zeros(9, dtype=np.float32) # initialize the state array
        
        # point mass model initial state
        self.state[0] = self.empty + self.fuel  # weight of the aircraft [lbf]
        self.state[1] = self.np_random.uniform(low=0, high=100, size=(1,))[0] # wind speed [knots]
        self.state[2] = self.np_random.uniform(low=300, high=545, size=(1,))[0] # cruise speed [knots]
        self.state[3] = self.np_random.uniform(low=400, high=500, size=(1,))[0] # target cruise speed based on research
        self.state[4] = self.np_random.uniform(low=30000, high=41000, size=(1,))[0] # altitude [ft]
        self.state[5] = self.np_random.uniform(low=33000, high=41000, size=(1,))[0] # target altitude based on research
        self.state[6] = self.np_random.uniform(low=35000, high=45000, size=(1,))[0] # fuel amount [lbs]
        self.state[7] = self.cruise_end # remaining distance to cruise end
        self.state[8] = self.np_random.uniform(low=5000, high=8000, size=(1,))[0] # thrust [lbf]
        
        return self._obs(), {}

    def step(self, action):
        # important constants
        fts_to_knots = 0.592484
        g = 32.2 # gravitational acceleration [ft/s^2]
        A = 120 # frontal area [ft^2]
        air_constant = 0.00339 # density and air constant
        TSFC = 0.5 # thrust specific fuel consumption [lbs/(lbf*hr)]
        
        # apply action to the environment
        self.state[8] = action[0]*500 # adjust thrust
        self.state[4] += action[1]*500 # adjust altitude

        # update drag and cruise speed
        self.drag = self.c_drag*(self.state[2] - self.state[1])**2 # update drag based on wind speed
        drag_force = self.drag*A*air_constant # drag force [lbf]
        accel = (self.state[8] - drag_force)/self.weight # acceleration
        self.state[2] += accel*self.dts*fts_to_knots # update cruise speed
        
        # adjust fuel and weight
        fuel_burned = abs(self.state[8]*TSFC*self.dt) # fuel burned in the current time step [lbs]
        self.state[6] -= fuel_burned # update remaining fuel [lbs]
        self.state[0] -= fuel_burned # update weight [lbs]
        self.state[1] = self.np_random.uniform(low=0, high=20, size=(1,))[0] # update wind speed in the state [knots]

        # update distance
        distance_traveled = self.state[2] * self.dt # distance traveled in the current time step [nautical miles]
        self.state[7] -= distance_traveled # update remaining distance to cruise end
            
        # calculate reward based on how close the current state is to the target state
        p = 1.2 # penalty factor
        fuel_used = p*(abs(self.state[8]*TSFC)) # amount of fuel used in step
        cruise_speed_diff = p*abs(self.state[2] - self.state[3]) # difference between current and target cruise speed
        altitude_diff = p*abs(self.state[4] - self.state[5]) # difference between current and target altitude
        stall_penalty = 0 # initialize stall penalty
        if self.state[2] < self.stall_speed:
            stall_penalty = p*50 # very large penalty for being below stall speed
        reward = - (cruise_speed_diff + altitude_diff + stall_penalty + fuel_used) # negative reward for being far from target
        print(f"Reward: {reward}, Cruise Speed Diff: {cruise_speed_diff}, Altitude Diff: {altitude_diff}, Stall Penalty: {stall_penalty}, Fuel Used: {fuel_used}")
        
        # initialize goal parameters
        terminated = False # initialize termination flag
        if self.state[7] >= self.cruise_end:
            terminated = True # mark the episode as terminated if cruise end is reached
        if self.state[6] <= 0:
            terminated = True # mark the episode as terminated if run out of fuel
        if self.state[4] >= 41000:
            terminated = True # mark the episode as terminated if above max altitude

        return self._obs(), reward, terminated, False, {"cost": reward}

    def _obs(self):
        return self.state

# check the environment
env = FlightEnv()
# check_env(env, warn=True) 
# print("No issues found")




# reset environment
obs, info = env.reset()
fuel_history = [] # assign fuel to an empty array
speed_history = [] # assign cruise speed to an empty array
alt_history = [] # assign altitude to an empty array
total_reward = 0 # initialize total reward

# random steps
for _ in range(1000):
  action = env.action_space.sample()  # random action
  obs, reward, terminated, truncated, info = env.step(action)
  
  weight, wind_speed, cruise_speed, target_cruise_speed, altitude, target_altitude, remaining_fuel, remaining_distance, thrust =  obs # assign variables to observations
  fuel_history.append(float(obs[6])) # get list of fuel over episode
  speed_history.append(float(obs[2])) # get list of cruise speed over episode
  alt_history.append(float(obs[4])) # get list of altitude over episode
  
  total_reward += reward # accumulate total reward

fig, axs = plt.subplots(1, 3, figsize=(6, 12))
axs[0].plot(speed_history, marker='o')
axs[0].set_xlabel("Step")
axs[0].set_ylabel("Speed Over Episode")

axs[1].plot(fuel_history, marker='x')
axs[1].set_xlabel("Step")
axs[1].set_ylabel("Fuel Over Episode")

axs[2].plot(alt_history, marker='s')
axs[2].set_xlabel("Step")
axs[2].set_ylabel("Altitude Over Episode")

plt.suptitle('Environment Performance', fontsize=16)
plt.tight_layout()
plt.show()
print("Total reward:", total_reward)
env.close()