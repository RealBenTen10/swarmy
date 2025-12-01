import math
import random
from swarmy.actuation import Actuation
import numpy as np


class MyController(Actuation):

    def __init__(self, agent, config):
        super().__init__(agent)
        self.config = config
        self.init_pos = True

        # parameters for gradient-following behaviour
        self.dt = 0.5  # integration step
        self.gradient_scale = 0.05  # scales the velocity magnitude
        self.max_velocity = 5.0  # clamp maximum displacement/velocity

    def controller(self):

        # Initialise robot position once
        if self.init_pos:
            W = self.config['world_width']
            H = self.config['world_height']
            start_x = random.randint(int(0.05 * W), int(0.2 * W))
            start_y = random.randint(0, H)

            # Set initial position and a random heading
            self.agent.set_position(start_x, start_y, random.randint(0, 360))
            self.init_pos = False

        x, y, heading_deg = self.agent.get_position()

        # Add current position to trajectory
        self.agent.trajectory.append((x, y))

        grad_x = self.agent.environment.getGradientX(x, y)
        grad_y = self.agent.environment.getGradientY(x, y)

        vx = grad_x * self.gradient_scale
        vy = grad_y * self.gradient_scale

        vel_mag = np.linalg.norm([vx, vy])
        if vel_mag > self.max_velocity:
            scale_factor = self.max_velocity / vel_mag
            vx *= scale_factor
            vy *= scale_factor

        dx = vx * self.dt
        dy = vy * self.dt

        new_x = x + dx
        new_y = y + dy

        if vel_mag > 1e-6:
            new_heading = math.degrees(math.atan2(dx, dy)) % 360
        else:
            new_heading = heading_deg  # Keep current heading if stationary

        self.agent.set_position(new_x, new_y, int(new_heading))

        self.stepForward(0)


    def torus(self):
        x, y, heading = self.agent.get_position()
        W = self.config['world_width']
        H = self.config['world_height']

        if x < 0: x = W
        if x > W: x = 0
        if y < 0: y = H
        if y > H: y = 0

        self.agent.set_position(x, y, int(heading))