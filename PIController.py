import numpy as np

class PIController:

    def __init__(self, kp: float = 0.18, ki: float = 0.045,  dt: float = 1/360, throttle_min: float = -1.0, throttle_max: float = 1.0,
    ):
        # gains
        self.kp = kp 
        self.ki = ki
        self.dt = dt

        # min and max throttle actions
        self.throttle_min = throttle_min
        self.throttle_max = throttle_max
        self.integral = 0.0 # initialize integral sum

    def reset(self, initial_throttle: float = 0.4):
        if self.ki > 0.0:
            self.integral = np.clip(initial_throttle, self.throttle_min, self.throttle_max) / self.ki
        else:
            self.integral = 0.0

    def predict(self, obs: np.ndarray, deterministic: bool = True):
       
        speed, target = float(obs[0]), float(obs[1]) # get speed and target from observation
        error = target - speed # calculate speed error

        # candidate integrator update
        candidate_integral = self.integral + error * self.dt
        u_unsat = self.kp * error + self.ki * candidate_integral

        # anti-windup clamping
        if self.u_min < u_unsat < self.u_max:
            self.integral = candidate_integral
        elif (u_unsat >= self.u_max and error < 0.0) or (u_unsat <= self.u_min and error > 0.0):
            # unwind integrator if error reverses direction
            self.integral = candidate_integral

        # calculate throttle and fuel consumption
        throttle = float(np.clip(self.kp * error + self.ki * self.integral, self.throttle_min, self.throttle_max))
        fuel_burned = abs(throttle*0.5*self.dt)

        # convert physical throttle [0, 1] -> normalized RL action [-1, 1]
        norm_action = np.array([2.0 * throttle - 1.0], dtype=np.float32)
        return norm_action, None