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
        self.dt = 0.5  # integration step (seconds)
        self.gradient_scale = 0.05  # scales the velocity magnitude
        self.max_velocity = 5.0  # clamp maximum displacement/velocity

    # -------------------------------------------------------------
    # TASK 3.2a — direct gradient controller
    # -------------------------------------------------------------
    def controller(self):

        # Initialise robot position once
        if self.init_pos:
            # Start robot in the high-potential (left) side
            W = self.config['world_width']
            H = self.config['world_height']
            # x is around 10% of width, y is random
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
        # Note: The gradient functions in my_world return P[i,j] - P[i+1,j], which is -∂P/∂x (moving from high to low)
        # However, the task formula is vx(t) = ∂P/∂x. Let's assume the goal is to move DOWN the potential.
        # Downhill movement requires velocity proportional to the NEGATIVE gradient: v ∝ -∇P
        # Our getGradientX is P[i,j] - P[i+1,j] ≈ -∂P/∂x. So, we use the raw gradient components.

        # P[i,j] - P[i+1,j] is the negative of the difference P[i+1,j] - P[i,j] (which is approx dx)
        # Let's use the provided approximation logic for simplicity, which moves the robot towards lower potential.

        # Approximate gradient components (P[i,j] - P[i+1,j] is the downhill direction in x)
        grad_x = self.agent.environment.getGradientX(x, y)
        grad_y = self.agent.environment.getGradientY(x, y)

        # 3. Calculate velocity components
        # v_x = ∂P/∂x, v_y = ∂P/∂y.
        # We assume the approximation functions in my_world.py are defined as the *negative* gradient to go downhill,
        # or we explicitly take the negative here to go downhill. Let's assume the gradient is the slope
        # and we want to move *down* the slope (negative gradient).

        # If grad_x is already a downhill measure, we use it directly:
        # velocity = slope * scale
        vx = grad_x * self.gradient_scale
        vy = grad_y * self.gradient_scale

        # 4. Clamp velocity magnitude
        vel_mag = np.linalg.norm([vx, vy])
        if vel_mag > self.max_velocity:
            scale_factor = self.max_velocity / vel_mag
            vx *= scale_factor
            vy *= scale_factor

        # 5. Calculate displacement
        dx = vx * self.dt
        dy = vy * self.dt

        # 6. Update position (holonomic control)
        new_x = x + dx
        new_y = y + dy

        # Update heading to face the direction of motion (optional, for visualization)
        if vel_mag > 1e-6:
            new_heading = math.degrees(math.atan2(dx, dy)) % 360
        else:
            new_heading = heading_deg  # Keep current heading if stationary

        self.agent.set_position(new_x, new_y, int(new_heading))

        # We keep the stepForward(0) to satisfy the Actuation base class requirement if needed,
        # but the position is already set.
        self.stepForward(0)

        # -------------------------------------------------------------

    # Torus world behaviour (optional)
    # -------------------------------------------------------------
    def torus(self):
        # This is not strictly required by the prompt, but good to keep.
        x, y, heading = self.agent.get_position()
        W = self.config['world_width']
        H = self.config['world_height']

        # wrap-around behaviour
        if x < 0: x = W
        if x > W: x = 0
        if y < 0: y = H
        if y > H: y = 0

        self.agent.set_position(x, y, int(heading))