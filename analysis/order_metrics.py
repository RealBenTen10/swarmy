import copy
from typing import Callable, Dict, List

import matplotlib.pyplot as plt
import numpy as np


def compute_order_parameter(velocities: List[np.ndarray]) -> float:
    """
    Compute global alignment order parameter va.
    velocities: list of 2D velocity vectors
    """
    if not velocities:
        return 0.0
    vel = np.array(velocities, dtype=float)
    norms = np.linalg.norm(vel, axis=1, keepdims=True)
    norms[norms == 0] = 1e-9
    v_hat = vel / norms
    mean_vec = v_hat.mean(axis=0)
    return float(np.linalg.norm(mean_vec))


def sweep_noise_levels(
    base_config: Dict,
    experiment_factory: Callable,
    eta_values: List[float],
    runs_per_eta: int = 5,
    average_window: int = 1000,
):
    """
    Run multiple flocking simulations across noise levels and return steady-state va.
    experiment_factory: function taking (config, step_callback) -> Experiment
    """
    results = {}
    for eta in eta_values:
        run_vals = []
        for _ in range(runs_per_eta):
            cfg = copy.deepcopy(base_config)
            cfg["noise_eta"] = eta
            order_trace: List[float] = []

            def step_cb(agent_list, _timestep):
                velocities = [getattr(a.actuation, "velocity", np.zeros(2)) for a in agent_list]
                order_trace.append(compute_order_parameter(velocities))

            exp = experiment_factory(cfg, step_cb)
            exp.run(cfg.get("rendering", -1), step_callback=step_cb)
            if order_trace:
                run_vals.append(float(np.mean(order_trace[-average_window:])))
        if run_vals:
            results[eta] = float(np.mean(run_vals))
    return results


def plot_order_vs_noise(results: Dict[float, float], output_path: str):
    etas = sorted(results.keys())
    vals = [results[e] for e in etas]
    plt.figure(figsize=(6, 4))
    plt.plot(etas, vals, marker="o")
    plt.xlabel("Noise η")
    plt.ylabel("Order parameter v_a")
    plt.ylim(0, 1.05)
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()

