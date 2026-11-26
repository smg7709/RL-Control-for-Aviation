# RL-Control-for-Aviation

This program implements a point mass model containing drag, thrust, pitch, speed, and altitude for a Boeing 737 Next Generation aircraft, which uses an RL agent and observations from the environment and targets to maximize fuel efficiency during the cruise portion of the flight.

Versions:

V1:
Standard PPO and RL parameters
Speed and Altitude control PI

V2:
PPO parameters changed to gamma = 0.999 and LR = 1e-4

V3:
PPO timesteps reduced to 250,000 to match SAC
PPO and SAC parameters made equal (gamma = 0.999, LR = 1e-4)
Pitch control added to action space for RL and PI
Decreased total fuel and distance, added reward/penalty constants for those termination conditions