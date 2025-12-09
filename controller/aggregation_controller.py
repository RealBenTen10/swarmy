import math
import random
from typing import List

import numpy as np
from swarmy.actuation import Actuation


class AggregationController(Actuation):
    """
    Simple aggregation with proximity stopping and timed waiting.
    Robots stop when a neighbor is inside sensing_radius, wait for
    wait_steps timesteps, then move away from local centroid or wander.
    """

    def __init__(self, agent, config):
        super().__init__(agent)
        self.config = config
        self.initialized = False

        agg_cfg = config.get("aggregation", {})
        self.sensing_radius = float(agg_cfg.get("sensing_radius", 20.0))
        self.wait_steps = int(agg_cfg.get("wait_steps", 8))
        self.move_speed = float(agg_cfg.get("move_speed", 1.5))
        self.dt = float(agg_cfg.get("dt", 1.0))

        self.wait_timer = 0
        self.velocity = np.zeros(2, dtype=float)

    def controller(self):
        if not self.initialized:
            self._initialize_pose()

        if self.wait_timer > 0:
            # stay stopped until timer expires
            self.wait_timer -= 1
            self.velocity[:] = 0.0
            return

        neighbors = self._get_neighbors_within(self.sensing_radius)
        if neighbors:
            # stop and start waiting
            self.wait_timer = self.wait_steps
            self.velocity[:] = 0.0
            return

        # move away from local centroid if any neighbors in sensing range, else wander
        away_vec = self._direction_from_local_centroid()
        if np.linalg.norm(away_vec) < 1e-6:
            # small random drive to keep motion
            angle = random.uniform(0, 2 * math.pi)
            away_vec = np.array([math.cos(angle), math.sin(angle)])

        desired = away_vec / (np.linalg.norm(away_vec) + 1e-9) * self.move_speed
        self.velocity = desired

        dx, dy = desired * self.dt
        x, y, _ = self.agent.get_position()
        # log trajectory for visualization
        self.agent.trajectory.append((x, y))
        new_x, new_y = x + dx, y + dy
        heading = math.degrees(math.atan2(dx, dy)) % 360
        self.agent.set_position(new_x, new_y, heading)

    def torus(self):
        x, y, heading = self.agent.get_position()
        W = self.config["world_width"]
        H = self.config["world_height"]
        x = x % W
        y = y % H
        self.agent.set_position(x, y, heading)

    # helpers
    def _initialize_pose(self):
        W = self.config["world_width"]
        H = self.config["world_height"]
        x = random.uniform(0, W)
        y = random.uniform(0, H)
        heading = random.uniform(0, 360)
        self.agent.set_position(x, y, heading)
        self.initialized = True

    def _get_neighbors_within(self, radius: float) -> List:
        neighbors = []
        for other in self.agent.environment.agentlist:
            if other is self.agent:
                continue
            ox, oy, _ = other.get_position()
            x, y, _ = self.agent.get_position()
            dist = math.hypot(ox - x, oy - y)
            if dist < radius:
                neighbors.append(other)
        return neighbors

    def _direction_from_local_centroid(self) -> np.ndarray:
        x, y, _ = self.agent.get_position()
        positions = []
        for other in self.agent.environment.agentlist:
            if other is self.agent:
                continue
            ox, oy, _ = other.get_position()
            if math.hypot(ox - x, oy - y) < self.sensing_radius:
                positions.append([ox, oy])
        if not positions:
            return np.zeros(2, dtype=float)
        centroid = np.mean(np.array(positions), axis=0)
        return np.array([x, y]) - centroid

