import numpy as np
import matplotlib.pyplot as plt
from Environment import FlightEnv
from PIController import PIController

class CustomRLAdapter:

    def __init__(self, policy_fn):
        self.policy_fn = policy_fn

    def reset(self, initial_throttle: float = 0.4):
        pass  # Add hidden-state reset here only if using an RNN/LSTM policy

    def predict(self, obs: np.ndarray, deterministic: bool = True):
        action = self.policy_fn(obs)
        action = np.asarray(action, dtype=np.float32).reshape(1,)
        return np.clip(action, -1.0, 1.0), None


def standardized_flight_profile(t: float):
    """Deterministic benchmark scenario: step commands + wind/climb disturbance."""
    if t < 12.0:
        v_target = 55.0
    elif t < 24.0:
        v_target = 65.0
    else:
        v_target = 48.0

    # Inject a +6 m/s wind gust and 3-degree climb angle at t >= 32s
    wind_gust = 6.0 if t >= 32.0 else 0.0
    gamma = np.deg2rad(3.0) if t >= 32.0 else 0.0
    return v_target, wind_gust, gamma


def evaluate_controller(name: str, controller, dt: float = 0.05) -> dict:
    env = FlightEnv(dt=dt, episode_seconds=42.0)
    env.set_eval_profile(standardized_flight_profile)
    obs, _ = env.reset(seed=50) # resets w/ seed 50
    controller.reset(initial_throttle=float(obs[4]))

    time, speed, target, throttle, rewards = [], [], [], [], [] # initialize empty lists for tracking
    done = False

    while not done:
        action, _ = controller.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated

        time.append(env.step_count * dt)
        speed.append(info["Cruise Speed"])
        target.append(info["Target"])
        throttle.append(info["Throttle"])
        rewards.append(reward)

    speed = np.array(speed)
    target = np.array(target)
    throttle = np.array(throttle)
    errors = target - speed

    # Step-response metrics on the t = 12s -> 24s step (55 m/s -> 65 m/s)
    idx_start, idx_end = int(12.0 / dt), int(24.0 / dt)
    step_v = speed[idx_start:idx_end]
    step_size = 65.0 - 55.0
    overshoot_pct = max(0.0, (np.max(step_v) - 65.0) / step_size * 100.0)

    # 2% settling time (+/- 0.2 m/s around 65 m/s)
    within_band = np.abs(step_v - 65.0) <= 0.02 * step_size
    settling_time = np.nan
    for i in range(len(within_band)):
        if np.all(within_band[i:]):
            settling_time = i * dt
            break

    return {
        "name": name,
        "time": np.array(time),
        "v_true": speed,
        "v_target": target,
        "throttle": throttle,
        "rmse": float(np.sqrt(np.mean(errors ** 2))),
        "mae": float(np.mean(np.abs(errors))),
        "overshoot_pct": float(overshoot_pct),
        "settling_time": float(settling_time),
        "chatter": float(np.sum(np.abs(np.diff(throttle)))),
        "total_reward": float(np.sum(rewards)),
    }


def main():
    dt = 1/360

    # 1. Tuned PI Baseline
    pi_agent = PIController(kp=0.18, ki=0.045, dt=dt)

    # 2. Plug in your trained custom PPO and SAC policies here:
    # Example with PyTorch:
    # ppo_agent = CustomRLAdapter(lambda obs: my_ppo_actor(torch.tensor(obs)).detach().numpy())
    # sac_agent = CustomRLAdapter(lambda obs: my_sac_actor.select_action(obs, eval=True))
    #
    # Placeholder stubs below so the script runs immediately out-of-the-box:
    ppo_agent = CustomRLAdapter(lambda obs: np.clip([0.35 * obs[2] - 0.15 * obs[3]], -1.0, 1.0))
    sac_agent = CustomRLAdapter(lambda obs: np.clip([0.40 * obs[2] - 0.18 * obs[3]], -1.0, 1.0))

    results = [
        evaluate_controller("PI Controller", pi_agent, dt=dt),
        evaluate_controller("Custom PPO", ppo_agent, dt=dt),
        evaluate_controller("Custom SAC", sac_agent, dt=dt),
    ]

    # Print comparison table
    header = f"{'Controller':<16} | {'RMSE (m/s)':>10} | {'Overshoot':>10} | {'Settle (s)':>10} | {'Chatter':>9} | {'Reward':>10}"
    print(header)
    print("-" * len(header))
    for r in results:
        print(
            f"{r['name']:<16} | {r['rmse']:>10.3f} | {r['overshoot_pct']:>9.2f}% | "
            f"{r['settling_time']:>10.2f} | {r['chatter']:>9.3f} | {r['total_reward']:>10.1f}"
        )

    # Plot tracking and actuator effort
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(11, 7), sharex=True)
    ax1.plot(results[0]["time"], results[0]["v_target"], "k--", linewidth=1.8, label="Target Airspeed")

    for r in results:
        ax1.plot(r["time"], r["v_true"], linewidth=1.6, label=r["name"])
        ax2.plot(r["time"], r["throttle"], linewidth=1.4, label=r["name"])

    ax1.axvline(32.0, color="gray", linestyle=":", label="Gust + Climb Disturbance")
    ax1.set_ylabel("Airspeed (m/s)")
    ax1.set_title("Airspeed Hold Benchmark: PI vs. PPO vs. SAC")
    ax1.legend(loc="upper left")
    ax1.grid(True, alpha=0.3)

    ax2.axvline(32.0, color="gray", linestyle=":")
    ax2.set_ylabel("Throttle [0, 1]")
    ax2.set_xlabel("Time (s)")
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()