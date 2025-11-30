from swarmy.environment import Environment
import pygame
import numpy as np


class My_environment(Environment):
    def __init__(self, config):
        self.config = config
        super().__init__(config)

        # Build potential field once during initialization
        self.definePotentialField()

    # -------------------------------------------------------------
    # Static walls
    # -------------------------------------------------------------
    def add_static_rectangle_object(self):
        # The existing static rectangles are fine to define the border for collision,
        # but the potential field is what defines the *repulsive* effect.
        #self.staticRectList.append(['BLACK', pygame.Rect(5, 5, self.config['world_width'] - 10, 5), 5])
        #self.staticRectList.append(['BLACK', pygame.Rect(5, 5, 5, self.config['world_height'] - 10), 5])
        #self.staticRectList.append(
        #    ['BLACK', pygame.Rect(5, self.config['world_height'] - 10, self.config['world_width'] - 10, 5), 5])
        #self.staticRectList.append(
        #    ['BLACK', pygame.Rect(self.config['world_width'] - 10, 5, 5, self.config['world_height'] - 10), 5])
        pass
    # -------------------------------------------------------------
    # Static circles (not used here)
    # -------------------------------------------------------------
    def add_static_circle_object(self):
        pass

    # -------------------------------------------------------------
    # Background rendering (draw potential field)
    # -------------------------------------------------------------
    def set_background_color(self):
        if hasattr(self, "P_surface"):
            surf = pygame.transform.scale(
                self.P_surface,
                (self.config['world_width'], self.config['world_height'])
            )
            self.displaySurface.blit(surf, (0, 0))
        else:
            self.displaySurface.fill(self.BACKGROUND_COLOR)

    def definePotentialField(self):
        """
        Create a potential field with:
        - a global slope left (high) → right (low)
        - several Gaussian obstacle hills (local maxima)
        - repulsive potential from top/bottom borders
        """

        width = self.config['world_width']
        height = self.config['world_height']

        # Parameters
        SLOPE_SCALE = 3000.0  # Stronger global slope
        num_obstacles = 12
        obstacle_height = 2000.0
        obstacle_sigma = 20.0  # Smaller, sharper obstacles

        # Border Repulsion Parameters
        # Strength of the repulsion
        BORDER_REPULSION_MAGNITUDE = 1000.0
        # Defines the distance over which the repulsion is significant (e.g., 50 pixels)
        REPULSION_INFLUENCE_DISTANCE = 50.0

        ys, xs = np.meshgrid(np.arange(height), np.arange(width), indexing='ij')

        # ------------------------------------------------------------------
        # 1. Global left→right slope (Attractive Component)
        # ------------------------------------------------------------------
        x_slope = np.linspace(1.0, 0.0, width)
        self.P = np.repeat(x_slope[np.newaxis, :], height, axis=0) * SLOPE_SCALE

        # ------------------------------------------------------------------
        # 2. Gaussian obstacles (Local Maxima/Repulsive)
        # ------------------------------------------------------------------
        rng = np.random.default_rng()

        for _ in range(num_obstacles):
            ox = rng.integers(0.1 * width, 0.9 * width)
            oy = rng.integers(0.1 * height, 0.9 * height)

            # Gaussian bump centered at (ox, oy)
            dist2 = (xs - ox) ** 2 + (ys - oy) ** 2
            bump = obstacle_height * np.exp(-dist2 / (2 * obstacle_sigma ** 2))

            self.P += bump

        # ------------------------------------------------------------------
        # 3. Top/Bottom Border Repulsion Potential (Local Maxima/Repulsive)
        # ------------------------------------------------------------------
        # Calculate distance to nearest border (top or bottom)
        dist_to_top = ys
        dist_to_bottom = height - 1 - ys

        # Minimum distance to any horizontal border
        dist_min = np.minimum(dist_to_top, dist_to_bottom)

        # Inverse function: increases sharply as dist_min approaches zero (i.e., near the border)
        # We cap the maximum influence distance to prevent infinite potential at the border.
        dist_min_safe = np.clip(dist_min, 1.0, REPULSION_INFLUENCE_DISTANCE)

        # Repulsive potential: proportional to 1 / distance^2 (or similar)
        # Note: Repulsion is only added if distance is less than influence distance
        repulsion_potential = np.where(
            dist_min < REPULSION_INFLUENCE_DISTANCE,
            BORDER_REPULSION_MAGNITUDE * (1 / dist_min_safe ** 2),
            0.0
        )

        self.P += repulsion_potential

        # ------------------------------------------------------------------
        # 4. Normalize to [0,1] for visualization
        # ------------------------------------------------------------------
        P_min, P_max = np.min(self.P), np.max(self.P)
        normP = (self.P - P_min) / (P_max - P_min + 1e-12)

        # ------------------------------------------------------------------
        # 5. Create red→blue RGB image (Red=High Potential, Blue=Low Potential)
        # ------------------------------------------------------------------
        R = (255 * normP).astype(np.uint8)
        G = np.zeros_like(R, dtype=np.uint8)
        B = (255 * (1.0 - normP)).astype(np.uint8)

        rgb = np.stack([R, G, B], axis=2)

        # pygame expects (width, height, 3)
        self.P_surface = pygame.surfarray.make_surface(np.transpose(rgb, (1, 0, 2)))

        return self.P

    def getPotential(self, x, y):
        xi = int(np.clip(x, 0, self.P.shape[1] - 1))
        yi = int(np.clip(y, 0, self.P.shape[0] - 1))
        return self.P[yi, xi]

    def getGradientX(self, x, y):
        # Ensures indices are valid before subtraction
        xi = int(np.clip(x, 0, self.P.shape[1] - 2))
        yi = int(np.clip(y, 0, self.P.shape[0] - 1))
        # P[i,j] - P[i+1,j] is the downhill direction in X
        return self.P[yi, xi] - self.P[yi, xi + 1]

    def getGradientY(self, x, y):
        # Ensures indices are valid before subtraction
        xi = int(np.clip(x, 0, self.P.shape[1] - 1))
        yi = int(np.clip(y, 0, self.P.shape[0] - 2))
        # P[i,j] - P[i+1,j] is the downhill direction in Y
        return self.P[yi, xi] - self.P[yi + 1, xi]