import math
import random
from typing import List

import numpy as np
from swarmy.actuation import Actuation


def _wrap_delta(dx: float, span: float) -> float:
    """Return shortest toroidal delta along one dimension."""
    half = span / 2.0
    if dx > half:
        return dx - span
    if dx < -half:
        return dx + span
    return dx


class FlockingController(Actuation):
    """
    Reynolds-style flocking with optional heading noise and toroidal wrapping.
    """

    def __init__(self, agent, config):
        super().__init__(agent)
        self.config = config
        flock_cfg = config.get("flocking", {})
        self.rs = float(flock_cfg.get("rs", 10.0))
        self.ra = float(flock_cfg.get("ra", 25.0))
        self.rc = float(flock_cfg.get("rc", 25.0))
        self.ws = float(flock_cfg.get("ws", 1.0))
        self.wa = float(flock_cfg.get("wa", 1.0))
        self.wc = float(flock_cfg.get("wc", 1.0))
        self.v0 = float(flock_cfg.get("v0", 1.0))
        self.noise_eta = float(flock_cfg.get("noise_eta", config.get("noise_eta", 0.0)))
        self.initialized = False
        self.velocity = np.zeros(2, dtype=float)

    def controller(self):
        if not self.initialized:
            self._initialize_pose()

        neighbors = self._neighbors()
        f_sep, f_ali, f_coh = self._compute_forces(neighbors)
        v_des = self.ws * f_sep + self.wa * f_ali + self.wc * f_coh
        if np.linalg.norm(v_des) < 1e-6:
            v_des = self.velocity if np.linalg.norm(self.velocity) > 0 else self._random_unit()

        v_des = self._apply_noise(v_des)
        v_des = v_des / (np.linalg.norm(v_des) + 1e-9) * self.v0
        self.velocity = v_des

        x, y, _ = self.agent.get_position()
        dx, dy = v_des
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
        # initialize with unit velocity along heading
        vx = math.cos(math.radians(heading)) * self.v0
        vy = math.sin(math.radians(heading)) * self.v0
        self.velocity = np.array([vx, vy])
        self.agent.set_position(x, y, heading)
        self.initialized = True

    def _neighbors(self) -> List:
        return [a for a in self.agent.environment.agentlist if a is not self.agent]

    def _compute_forces(self, neighbors: List):
        pos_i = np.array(self.agent.get_position()[:2])
        sep = np.zeros(2)
        ali = np.zeros(2)
        coh_acc = np.zeros(2)
        n_align = 0
        n_coh = 0

        W = self.config["world_width"]
        H = self.config["world_height"]

        for n in neighbors:
            pos_j = np.array(n.get_position()[:2])
            d = np.array([
                _wrap_delta(pos_j[0] - pos_i[0], W),
                _wrap_delta(pos_j[1] - pos_i[1], H),
            ])
            dist = np.linalg.norm(d)
            if dist < 1e-9:
                continue
            if dist < self.rs:
                sep -= d / (dist ** 2)
            if dist < self.ra:
                ali += getattr(n.actuation, "velocity", np.zeros(2))
                n_align += 1
            if dist < self.rc:
                coh_acc += pos_i + d  # already wrapped toward neighbor
                n_coh += 1

        if n_align > 0:
            ali = ali / n_align - self.velocity
        if n_coh > 0:
            center = coh_acc / n_coh
            coh = center - pos_i
        else:
            coh = np.zeros(2)

        return sep, ali, coh

    def _apply_noise(self, v: np.ndarray) -> np.ndarray:
        if self.noise_eta <= 0:
            return v
        angle = math.atan2(v[1], v[0])
        noise = random.uniform(-self.noise_eta / 2.0, self.noise_eta / 2.0)
        angle += noise
        speed = np.linalg.norm(v)
        return np.array([speed * math.cos(angle), speed * math.sin(angle)])

    @staticmethod
    def _random_unit() -> np.ndarray:
        ang = random.uniform(0, 2 * math.pi)
        return np.array([math.cos(ang), math.sin(ang)])

