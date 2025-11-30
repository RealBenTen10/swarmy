import math
import random
from swarmy.actuation import Actuation
import numpy as np


class MyController2(Actuation):

    def __init__(self, agent, config):
        super().__init__(agent)
        self.config = config
        self.init_pos = True

        # parameters for indirect control (momentum-based)
        self.dt = 0.5  # integration step (seconds)
        self.momentum_c = 0.8  # Momentum factor c (0 < c << 1 is specified, but
        # a value close to 1 is typical for momentum/memory)
        self.gradient_scale = 0.005  # Scales the acceleration magnitude (must be smaller now)
        self.max_velocity = 1.0  # Clamp maximum velocity (allows faster movement)

        # New state variables to store previous velocity (robot's memory)
        self.vx_prev = 0.0
        self.vy_prev = 0.0

    # -------------------------------------------------------------
    # TASK 3.2b — indirect gradient controller (Momentum/Inertia)
    # -------------------------------------------------------------
    def controller(self):

        # Initialise robot position once (same as before)
        if self.init_pos:
            # Start robot in the high-potential (left) side
            W = self.config['world_width']
            H = self.config['world_height']
            start_x = random.randint(int(0.05 * W), int(0.2 * W))
            start_y = random.randint(0, H)

            # Set initial position and a random heading
            self.agent.set_position(start_x, start_y, random.randint(0, 360))
            self.init_pos = False

        # 1. Read robot state in pixel coordinates
        x, y, heading_deg = self.agent.get_position()

        # Add current position to trajectory
        self.agent.trajectory.append((x, y))

        # 2. Get gradient of potential field at robot location
        # These represent the components of the vector pointing DOWN the slope (towards lower potential)
        grad_x = self.agent.environment.getGradientX(x, y)
        grad_y = self.agent.environment.getGradientY(x, y)

        # Scale the gradient to prevent instantaneous huge acceleration
        accel_x = grad_x * self.gradient_scale
        accel_y = grad_y * self.gradient_scale

        # 3. Calculate NEW velocity components (Momentum + Acceleration)
        # v(t) = c * v(t-Δt) + ∂P/∂x
        # We use a scaled version of the previous velocity to ensure units are consistent
        vx_new = self.momentum_c * self.vx_prev + accel_x
        vy_new = self.momentum_c * self.vy_prev + accel_y

        # 4. Normalise the new velocity vector v/|v|
        # This prevents runaway speed increase and enforces the maximum velocity.
        vel_mag = np.linalg.norm([vx_new, vy_new])

        if vel_mag < 1e-6:
            # Robot is stationary
            vx = 0.0
            vy = 0.0
        else:
            # Normalize to maximum velocity
            scale_factor = self.max_velocity / vel_mag
            vx = vx_new * scale_factor
            vy = vy_new * scale_factor

        # 5. Calculate displacement
        dx = vx * self.dt
        dy = vy * self.dt

        # 6. Update internal state (for next iteration)
        self.vx_prev = vx
        self.vy_prev = vy

        # 7. Update position (holonomic control)
        new_x = x + dx
        new_y = y + dy

        # Update heading to face the direction of motion (optional, for visualization)
        if vel_mag > 1e-6:
            new_heading = math.degrees(math.atan2(dx, dy)) % 360
        else:
            new_heading = heading_deg  # Keep current heading if stationary

        self.agent.set_position(new_x, new_y, int(new_heading))

        self.stepForward(0)

    # -------------------------------------------------------------
    # Torus world behaviour (optional)
    # -------------------------------------------------------------
    def torus(self):
        # This is kept from the previous code
        x, y, heading = self.agent.get_position()
        W = self.config['world_width']
        H = self.config['world_height']

        # wrap-around behaviour
        if x < 0: x = W
        if x > W: x = 0
        if y < 0: y = H
        if y > H: y = 0

        self.agent.set_position(x, y, int(heading))