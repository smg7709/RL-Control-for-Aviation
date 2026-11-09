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
        self.empty = 110000 # weight of the aircraft [lbf]
        self.fuel = 40000 # weight of fuel [lbf]
        self.weight = self.empty + self.fuel # total aircraft weight [lbf]
        self.stall_speed = 128 # stall speed [knots]
        self.thrust = 5000 # initial thrust [lbf]
        self.min_altitude = 30000 # minimum altitude [ft]
        self.max_altitude = 41000 # maximum altitude [ft]
        self.min_speed = 300 # minimum cruise speed [knots]
        self.max_speed = 545 # maximum cruise speed [knots]
        self.min_thrust = 2500 # minimum cruise thrust [lbf]
        self.max_thrust = 12000 # maximum cruise thrust [lbf]
        self.area = 1344 # frontal area [ft^2]

        # initialize simulation parameters
        self.dt = 1/360 # time step [hr]
        self.dts = self.dt*3600 # time step [s]
        self.cruise_start = 0
        self.cruise_end = self.np_random.uniform(low=500, high=5000, size=(1,))[0] # total cruise distance [nautical miles]
        self.speed_delta = 1 # maximum change in cruise speed per time step [knots]
        self.altitude_delta = 500 # maximum change in altitude per time step [ft]
        self.c_drag = 0.035 # coefficient of drag [-]
        self.air_density = 0.0006 # air density at cruise altitude [slugs/ft^3]
        
        # initialize actions and observations
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32) # continuous action space for cruise speed and altitude adjustments
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(9,), dtype=np.float32)  # weight, wind speed, cruise speed, target cruise speed, altitude, target altitude, remaining fuel, remaining distance to cruise end, thrust

    def reset(self, seed=None, options=None):
        super().reset(seed=seed) # reset the environment with the given seed
        self.state = np.zeros(9, dtype=np.float32) # initialize the state array
        
        # point mass model initial state
        self.state[0] = self.empty + self.fuel  # weight of the aircraft [lbf]
        self.state[1] = self.np_random.uniform(low=0, high=100, size=(1,))[0] # wind speed [knots]
        self.state[2] = self.np_random.uniform(low=self.min_speed, high=self.max_speed, size=(1,))[0] # cruise speed [knots]
        
        # determine airspeed and drag based on new initial speed, prevents steep decline in airspeed when new episode starts
        airspeed = (self.state[2] - self.state[1])*1.68781 # airspeed [ft/s]
        self.drag = 0.5*self.air_density*self.c_drag*self.area*airspeed**2 # calculate aerodynamic drag based on airspeed [lbf]

        self.state[3] = self.np_random.uniform(low=400, high=500, size=(1,))[0] # target cruise speed based on research
        self.state[4] = self.np_random.uniform(low=self.min_altitude, high=self.max_altitude, size=(1,))[0] # altitude [ft]
        self.state[5] = self.np_random.uniform(low=33000, high=self.max_altitude, size=(1,))[0] # target altitude based on research
        self.state[6] = self.np_random.uniform(low=35000, high=45000, size=(1,))[0] # fuel amount [lbf]
        self.state[7] = self.cruise_end # remaining distance to cruise end
        self.state[8] = float(np.clip(self.drag, self.min_thrust, self.max_thrust)) # update thrust in the state [lbf]

        
        return self._obs(), {}

    def step(self, action):
        # important constants
        fts_to_knots = 0.592484
        knots_to_fts = 1.68781
        g = 32.2 # gravitational acceleration [ft/s^2]
        TSFC = 0.5 # thrust specific fuel consumption [lbs/(lbf*hr)]
        
        # apply action to the environment
        new_speed = float(np.clip(self.state[2] + action[0]*self.speed_delta, self.min_speed, self.max_speed)) # determine new cruise speed [knots]
        self.state[4] = float(np.clip(self.state[4] + action[1]*self.altitude_delta, self.min_altitude, self.max_altitude)) # adjust altitude [ft]

        # calculate required acceleration, drag, thrust
        accel_req = (new_speed - self.state[2])*knots_to_fts/self.dts # required acceleration to reach new speed [ft/s^2]
        airspeed = (self.state[2] - self.state[1])*knots_to_fts # airspeed [ft/s]
        self.drag = 0.5*self.air_density*self.c_drag*self.area*airspeed**2 # calculate aerodynamic drag based on airspeed [lbf]
        thrust_req = self.drag + self.state[0]*accel_req/g # required thrust to overcome drag and achieve acceleration [lbf]

        # update thrust and speed
        self.state[8] = float(np.clip(thrust_req, self.min_thrust, self.max_thrust)) # update thrust in the state [lbf]
        accel = (self.state[8] - self.drag)*g/self.state[0] # calculate acceleration based on thrust and drag [ft/s^2]
        self.state[2] = float(np.clip(self.state[2] + accel*self.dts*fts_to_knots, self.min_speed, self.max_speed)) # update cruise speed in the state [knots]

        # update fuel and weight
        fuel_burned = abs(self.state[8]*TSFC*self.dt) # fuel burned in the current time step [lbs]
        self.state[6] -= fuel_burned # update remaining fuel [lbs]
        self.state[0] -= fuel_burned # update weight [lbs]
        self.state[1] = self.np_random.uniform(low=0, high=20, size=(1,))[0] # update wind speed in the state [knots]

        # update distance
        distance_traveled = self.state[2] * self.dt # distance traveled in the current time step [nautical miles]
        self.state[7] -= distance_traveled # update remaining distance to cruise end

        # gently update target cruise speed and altitude
        self.state[3] = float(np.clip(self.state[3] + self.np_random.uniform(-0.5, 0.5), 400, 500))
        self.state[5] = float(np.clip(self.state[5] + self.np_random.uniform(-50, 50), 33000, 39000))

        # calculate reward based on how close the current state is to the target state
        fuel_used = (abs(self.state[8]*TSFC)) # amount of fuel used in step
        cruise_speed_diff = abs(self.state[2] - self.state[3])/100 # difference between current and target cruise speed
        altitude_diff = abs(self.state[4] - self.state[5])/1000 # difference between current and target altitude
        fuel_penalty = fuel_used/1000 # penalty for fuel used
        
        stall_penalty = 0 # initialize stall penalty
        if self.state[2] < self.stall_speed:
            stall_penalty = 5 # large penalty for being below stall speed

        reward = -(cruise_speed_diff + 0.5*altitude_diff + fuel_penalty + stall_penalty) # total reward is negative sum of penalties
            # altitude is less important than cruise speed and fuel, so it is weighted less in the reward function

        # initialize goal parameters
        terminated = False # initialize termination flag
        if self.state[7] >= self.cruise_end:
            terminated = True # mark the episode as terminated if cruise end is reached
            reward = 10000 # large reward for reaching cruise end
        if self.state[6] <= 0:
            terminated = True # mark the episode as terminated if run out of fuel

        return self._obs(), reward, terminated, False, {"cost": reward}

    def _obs(self):
        return self.state

# check the environment
env = FlightEnv()
# check_env(env, warn=True) 
# print("No issues found")

#random sample testing

if __name__ == "__main__":
    env=FlightEnv()

    # reset environment
    obs, info = env.reset()
    fuel_history = [] # assign fuel to an empty array
    speed_history = [] # assign cruise speed to an empty array
    target_speed_history = [] # assign target cruise speed to an empty array
    alt_history = [] # assign altitude to an empty array
    target_alt_history = [] # assign target altitude to an empty array
    total_reward = 0 # initialize total reward

    # random steps
    for _ in range(1000):
        # action = env.action_space.sample()  # random action
        action = [1, 0]  # increase cruise speed, maintain altitude
        obs, reward, terminated, truncated, info = env.step(action)
        
        weight, wind_speed, cruise_speed, target_cruise_speed, altitude, target_altitude, remaining_fuel, remaining_distance, thrust =  obs # assign variables to observations
        fuel_history.append(float(obs[6])) # get list of fuel over episode
        speed_history.append(float(obs[2])) # get list of cruise speed over episode
        alt_history.append(float(obs[4])) # get list of altitude over episode
        target_speed_history.append(float(obs[3])) # get list of target cruise speed over episode
        target_alt_history.append(float(obs[5])) # get list of target altitude over episode

        total_reward += reward # accumulate total reward

        a_speed = np.clip((target_cruise_speed - cruise_speed) / 100, -1, 1)
        a_alt   = np.clip((target_altitude - altitude) / 1000, -1, 1)
        action_reward = np.array([a_speed, a_alt], dtype=np.float32)
        print(action_reward)
            

    fig, axs = plt.subplots(1, 3, figsize=(6, 12))
    axs[0].plot(target_speed_history, label="target")
    axs[0].plot(speed_history, label="speed")
    axs[0].legend()
    axs[0].set_xlabel("Step")
    axs[0].set_ylabel("Speed Over Episode")

    axs[1].plot(fuel_history, marker='x')
    axs[1].set_xlabel("Step")
    axs[1].set_ylabel("Fuel Over Episode")

    axs[2].plot(target_alt_history, label="target")
    axs[2].plot(alt_history, label="altitude")
    axs[2].legend()
    axs[2].set_xlabel("Step")
    axs[2].set_ylabel("Altitude Over Episode")

    plt.suptitle('Environment Performance', fontsize=16)
    plt.tight_layout()
    plt.show()
    print("Total reward:", total_reward)
    env.close()