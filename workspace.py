# =============================================================================
# created by:   Samer Al-Magazachi
# created on:   06/04/2021 -- 13/04/2022
# version:      0.9
# status:       prototype
# =============================================================================
import os
import yaml
from swarmy.experiment import Experiment
from analysis.order_metrics import plot_order_vs_noise, sweep_noise_levels

# load the configuration file, check the config.yaml file for more information and to change to your needs
with open("config.yaml", "r") as file:
    config = yaml.load(file, Loader=yaml.FullLoader)

# Import implementations of the controller, sensor, environment and agent
from controller.aggregation_controller import AggregationController
from controller.flocking_controller import FlockingController
from sensors.bumper_sensor import BumperSensor
from world.my_world import My_environment
from agent.my_agent import MyAgent


def build_experiment(local_config):
    agent_controller = [AggregationController]
    if local_config.get("experiment_type") in ["flocking", "flocking_noise"]:
        agent_controller = [FlockingController]
    agent_sensing = []
    return Experiment(local_config, agent_controller, agent_sensing, My_environment, MyAgent)


def run_single_mode():
    exp = build_experiment(config)
    exp.run(config.get("rendering", 1))


def run_noise_sweep():
    sweep_cfg = config.get("noise_sweep", {})
    etas = sweep_cfg.get("eta_values", [0.0, 0.5, 1.0])
    runs_per_eta = sweep_cfg.get("runs_per_eta", 5)
    avg_window = sweep_cfg.get("average_window", 1000)
    output_plot = sweep_cfg.get("output_plot", "plots/order_vs_noise.png")
    os.makedirs(os.path.dirname(output_plot), exist_ok=True)

    def factory(cfg, step_cb):
        cfg["experiment_type"] = "flocking"
        cfg["rendering"] = cfg.get("rendering", -1)
        return build_experiment(cfg)

    results = sweep_noise_levels(
        config,
        factory,
        eta_values=etas,
        runs_per_eta=runs_per_eta,
        average_window=avg_window,
    )
    plot_order_vs_noise(results, output_plot)
    print(f"Saved order-vs-noise plot to {output_plot}")


if __name__ == "__main__":
    mode = config.get("experiment_type", "aggregation")
    if mode == "flocking_noise":
        run_noise_sweep()
    else:
        run_single_mode()
