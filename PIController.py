import numpy as np

class PIController:

    def __init__(self, speed_delta: float, 
                 altitude_delta: float, 
                 pitch_delta: float,
                 kps: float = 0.5, 
                 kis: float = 0.005, 
                 kpa: float = 0.5, 
                 kia: float = 0,
                 kpp: float = 0.5,
                 kip: float = 0,
                 trim_pitch: float = 3,
                 dt: float = 1/360,):
        
        # gains scaled
        self.kps = kps/speed_delta
        self.kis = kis/speed_delta
        self.kpa = kpa/altitude_delta
        self.kia = kia/altitude_delta
        self.kpp = kpp/pitch_delta
        self.kip = kip/pitch_delta

        # constants defined with self
        self.trim_pitch = trim_pitch
        self.dt = dt
        
        # initialize integrals
        self.int_s = 0
        self.int_a = 0
        self.int_p = 0

    def reset(self):
        self.int_s = 0
        self.int_a = 0
        self.int_p = 0

    def step_loop(self, error: float, integral: float, kp: float, ki: float):
        # include anti-windup clamping
        candidate_int = integral + error*self.dt
        u_unsat = kp*error + ki*candidate_int

        # integrate if unsaturated
        if (-1 < u_unsat < 1) or (u_unsat >= 1 and error < 0) or (u_unsat <= -1 and error > 0):
            integral = candidate_int

        u = float(np.clip(kp*error + ki*integral, -1, 1))
        return u, integral
        
    def predict(self, state: np.ndarray, deterministic = True):
       # get raw states from environment
        speed, target_speed = float(state[2]), float(state[3])
        altitude, target_altitude = float(state[4]), float(state[5])
        pitch = float(state[9])

        # calculate errors
        speed_err = target_speed - speed
        altitude_err = target_altitude - altitude
        pitch_err = self.trim_pitch - pitch

        # speed change
        speed_action, self.int_s = self.step_loop(speed_err, self.int_s, self.kps, self.kis)

        # altitude change
        altitude_action, self.int_a = self.step_loop(altitude_err, self.int_a, self.kpa, self.kia)

        # pitch change
        pitch_action, self.int_p = self.step_loop(pitch_err, self.int_p, self.kpp, self.kip)

        # return actions
        action = np.array([speed_action, altitude_action, pitch_action], dtype=np.float32)
        return action, None