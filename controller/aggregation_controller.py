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
        self.min_separation = float(agg_cfg.get("min_separation", 15.0))  # minimum distance before stopping

        self.wait_timer = 0
        self.velocity = np.zeros(2, dtype=float)
        self.last_heading = None  # store last heading to prevent rapid changes

    def controller(self):
        if not self.initialized:
            self._initialize_pose()
            x, y, h = self.agent.get_position()
            self.last_heading = h

        if self.wait_timer > 0:
            # stay stopped until timer expires, maintain current heading
            self.wait_timer -= 1
            self.velocity[:] = 0.0
            x, y, h = self.agent.get_position()
            self.agent.trajectory.append((x, y))
            return

        # Check for neighbors within minimum separation distance
        neighbors = self._get_neighbors_within(self.min_separation)
        if neighbors:
            # stop and start waiting, keep current heading
            self.wait_timer = self.wait_steps
            self.velocity[:] = 0.0
            x, y, h = self.agent.get_position()
            self.agent.trajectory.append((x, y))
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
        
        # Fix: atan2 takes (y, x) not (x, y)
        new_heading = math.degrees(math.atan2(dy, dx)) % 360
        
        # Smooth heading changes to prevent rapid spinning
        if self.last_heading is not None:
            # Calculate shortest angular distance
            diff = (new_heading - self.last_heading + 180) % 360 - 180
            # Limit maximum change per step to prevent spinning
            max_change = 30.0  # degrees per step
            if abs(diff) > max_change:
                new_heading = (self.last_heading + max_change * np.sign(diff)) % 360
        
        self.last_heading = new_heading
        self.agent.set_position(new_x, new_y, int(new_heading))

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
        self.agent.set_position(x, y, int(heading))
        self.last_heading = float(heading)
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

