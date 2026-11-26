# initialize open ai gym and other tools
import gymnasium as gym
import numpy as np
from gymnasium import spaces
from stable_baselines3.common.env_checker import check_env
import matplotlib.pyplot as plt

class FlightEnv(gym.Env):
    def __init__(self):
        super(FlightEnv, self).__init__()

        # initialize aircraft parameters
        self.empty = 110000 # weight of the aircraft [lbf]
        self.area = 1344 # frontal area [ft^2]
        self.c_drag = 0.035 # coefficient of drag [-]

        self.min_altitude = 30000 # minimum altitude [ft]
        self.max_altitude = 41000 # maximum altitude [ft]

        self.min_speed = 128 # stall speed [knots]
        self.max_speed = 545 # maximum speed [knots]

        self.min_thrust = 2500 # minimum cruise thrust [lbf]
        self.max_thrust = 12000 # maximum cruise thrust [lbf]

        self.min_pitch = -15 # minimum pitch angle [deg]
        self.max_pitch = 20 # maximum pitch angle [deg]
        self.trim_pitch = 3 # trim pitch angle [deg]

        self.min_fuel = 15000 # minimum fuel weight [lbf]
        self.max_fuel = 25000 # maximum fuel weight [lbf]

        # initialize simulation parameters
        self.dt = 1/360 # time step [hr]
        self.dts = self.dt*3600 # time step [s]

        self.speed_delta = 1 # maximum change in cruise speed per time step [knots]
        self.altitude_delta = 250 # maximum change in altitude per time step [ft]
        self.pitch_delta = 2 # maximum change in pitch angle per time step [deg]

        self.step_count = 0 # number of steps

        # conversion factors
        self.fts_to_knots = 0.592484
        self.knots_to_fts = 1.68781
        self.deg_to_rad = np.pi/180

        
        # initialize actions and observations
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(3,), dtype=np.float32) # continuous action space for speed, altitude, pitch adjustments
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(10,), dtype=np.float32)  # weight, wind speed, cruise speed, target cruise speed, altitude, target altitude, remaining fuel, remaining distance, thrust, pitch angle

    def edge_penalty(self, value, low, high, margin_frac=0.1): 
        margin = margin_frac*(high-low) # 10% of total range
        if value < low + margin:
            return (low + margin - value)/margin
        if value > high - margin:
            return (value - (high - margin))/margin
        return 0.0

    def reset(self, seed=None, options=None):
        super().reset(seed=seed) # reset the environment with the given seed
        self.step_count = 0 # number of steps
        self.state = np.zeros(10, dtype=np.float32) # initialize the state array
        
        # point mass model initial state
        self.state[0] = self.empty + self.np_random.uniform(low=self.min_fuel, high=self.max_fuel, size=(1,))[0]  # weight of the aircraft [lbf]
        self.state[1] = self.np_random.uniform(low=0, high=30, size=(1,))[0] # wind speed [knots]
        self.state[2] = self.np_random.uniform(low=self.min_speed, high=self.max_speed, size=(1,))[0] # cruise speed [knots]
        
        # determine airspeed and drag based on new initial speed, prevents steep decline in airspeed when new episode starts
        airspeed = (self.state[2] - self.state[1])*self.knots_to_fts # airspeed [ft/s]
        density = 0.002377 * np.exp(-self.state[4] / 29900) # density based on current altitude [lb/ft^3]
        pitch_diff = self.deg_to_rad*(self.state[9]-self.trim_pitch) # pitch difference [rad]
        c_drag_eff = self.c_drag*(1.05*(pitch_diff)**2) # effective coefficient of drag [-], dependent on current pitch angle and trim angle
        self.drag = 0.5*density*c_drag_eff*self.area*airspeed**2 # calculate aerodynamic drag based on airspeed [lbf]

        self.state[3] = self.np_random.uniform(low=400, high=500, size=(1,))[0] # target cruise speed [knots]
        self.state[4] = self.np_random.uniform(low=self.min_altitude, high=self.max_altitude, size=(1,))[0] # cruising altitude [ft]
        self.state[5] = self.np_random.uniform(low=33000, high=self.max_altitude, size=(1,))[0] # target cruising altitude [ft]
        self.state[6] = self.np_random.uniform(low=self.min_fuel, high=self.max_fuel, size=(1,))[0] # fuel amount [lbf]
        self.state[7] = self.np_random.uniform(low=1000, high=3500, size=(1,))[0] # cruise distance [nautical miles]
        self.state[8] = float(np.clip(self.drag, self.min_thrust, self.max_thrust)) # thrust [lbf]
        self.state[9] = self.np_random.uniform(low=self.min_pitch, high=self.max_pitch, size=(1,))[0] # pitch angle [deg]
        
        return self._obs(), {}

    def step(self, action):
        # important constants
        g = 32.2 # gravitational acceleration [ft/s^2]
        TSFC = 0.5 # thrust specific fuel consumption [lbs/(lbf*hr)]
        density = 0.002377 * np.exp(-self.state[4] / 29900) # density based on current altitude [lb/ft^3]
        self.step_count += 1 # update step count
        
        # apply action to the environment
        self.state[9] = float(np.clip(self.state[9] + action[2]*self.pitch_delta, self.min_pitch, self.max_pitch)) # update pitch angle [deg]
        new_speed = float(np.clip(self.state[2] + action[0]*self.speed_delta, self.min_speed, self.max_speed)) # determine new cruise speed [knots]
        self.state[4] = float(np.clip(self.state[4] + action[1]*self.altitude_delta, self.min_altitude, self.max_altitude)) # update altitude [ft]

        # calculate required acceleration and airspeed
        accel_req = (new_speed - self.state[2])*self.knots_to_fts/self.dts # required acceleration to reach new speed [ft/s^2]
        airspeed = (self.state[2] - self.state[1])*self.knots_to_fts # airspeed [ft/s]
        
        # calculate drag and thrust required to overcome drag
        pitch_diff = self.deg_to_rad*(self.state[9]-self.trim_pitch) # pitch difference [rad]
        c_drag_eff = self.c_drag*(1.05*(pitch_diff)**2) # effective coefficient of drag [-], dependent on current pitch angle and trim angle
        self.drag = 0.5*density*c_drag_eff*self.area*airspeed**2 # calculate aerodynamic drag based on airspeed [lbf]
        thrust_req = self.drag + self.state[0]*accel_req/g # required thrust to overcome drag and achieve acceleration [lbf]

        # update thrust and speed
        self.state[8] = float(np.clip(thrust_req, self.min_thrust, self.max_thrust)) # update thrust [lbf]
        accel = (self.state[8] - self.drag)*g/self.state[0] # calculate acceleration based on thrust and drag [ft/s^2]
        self.state[2] = float(np.clip(self.state[2] + accel*self.dts*self.fts_to_knots, self.min_speed, self.max_speed)) # update cruise speed [knots]

        # update fuel, weight, wind speed, distance
        fuel_burned = abs(self.state[8]*TSFC*self.dt) # fuel burned in the current time step [lbf]
        self.state[6] -= fuel_burned # update remaining fuel [lbf]
        self.state[0] -= fuel_burned # update weight [lbf]
        self.state[1] = float(np.clip(self.state[1] + self.np_random.uniform(-0.5, 0.5), 0, 30)) # nudge wind speed [knots]
        self.state[7] -= self.state[2] * self.dt # update remaining distance [nautical miles]

        # update target cruise speed and altitude every 10 steps (to mimic ATC commands)
        if self.step_count % 10 == 0:
            self.state[3] = float(np.clip(self.state[3] + self.np_random.uniform(-5, 5), 400, 500)) # target cruise speed [knots]
            self.state[5] = float(np.clip(self.state[5] + self.np_random.uniform(-200, 200), 33000, 39000)) # target altitude [knots]


        # REWARD EQUATION ----------------------------------------------------------------------------------------------------
        terminated = False # initialize termination flag
        
        speed_diff = abs(self.state[3] - self.state[2])/20 # target speed - current speed
        altitude_diff = abs(self.state[5] - self.state[4])/5000 # target altitude - current altitude

        fuel_penalty  = fuel_burned / (self.max_thrust*TSFC*self.dt) # fuel used in step, normalized by max fuel usage per step

        speed_edge = self.edge_penalty(self.state[2], self.min_speed, self.max_speed) # penalty for being near the edge of the speed range
        alt_edge   = self.edge_penalty(self.state[4], self.min_altitude, self.max_altitude) # penalty for being near the edge of the altitude range

        # initialize fuel and distance constant rewards
        dist_reward = 0
        zerofuel_penalty = 0

        # check if cruise end reached
        if self.state[7] <= 0:
            dist_reward = 500 # large reward for reaching end of cruise
            terminated = True # end episode
        
        # check if fuel has run out
        if self.state[6] <= 0:
            zerofuel_penalty = -500 # large penalty for running out of fuel
            terminated = True # end episode

        # total reward is the negative sum of 5 weighted components plus fuel and distance constants
        reward = -(0.5 * speed_diff + 0.5 * altitude_diff + 2 * fuel_penalty + 0.5 * speed_edge + 0.5 * alt_edge) + dist_reward + zerofuel_penalty

        return self._obs(), reward, terminated, False, {"cost": reward}

    def _obs(self):
        return np.array([
            self.state[0], # weight [lbf]
            self.state[1], # wind [knots]
            self.state[2], # cruise speed [knots]
            (self.state[3] - self.state[2]), # cruise speed error (target - current)
            self.state[4], # altitude [feet]
            (self.state[5] - self.state[4]), # altitude error (target - current)
            self.state[6], # fuel [lbs]
            self.state[7], # remaining distance [nautical miles]
            self.state[8], # thrust [lbf]
            self.state[9], # pitch angle [deg]
        ], dtype=np.float32)

# check the environment
# env = FlightEnv()
# check_env(env, warn=True) 
# print("No issues found")