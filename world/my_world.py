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
        self.staticRectList.append(['BLACK', pygame.Rect(5, 5, self.config['world_width'] - 10, 5), 5])
        self.staticRectList.append(['BLACK', pygame.Rect(5, 5, 5, self.config['world_height'] - 10), 5])
        self.staticRectList.append(['BLACK', pygame.Rect(5, self.config['world_height'] - 10, self.config['world_width'] - 10, 5), 5])
        self.staticRectList.append(['BLACK', pygame.Rect(self.config['world_width'] - 10, 5, 5, self.config['world_height'] - 10), 5])


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
        - several Gaussian obstacle hills
        """

        width = self.config['world_width']
        height = self.config['world_height']

        # Gradient scale to make it meaningful for controller
        # INCREASED SLOPE_SCALE for a stronger global slope
        SLOPE_SCALE = 3000.0

        # ------------------------------------------------------------------
        # 1. Global left→right slope
        # ------------------------------------------------------------------
        x_slope = np.linspace(1.0, 0.0, width)
        self.P = np.repeat(x_slope[np.newaxis, :], height, axis=0) * SLOPE_SCALE

        # ------------------------------------------------------------------
        # 2. Gaussian obstacles (local maxima)
        # ------------------------------------------------------------------
        num_obstacles = 12
        obstacle_height = 2000.0  # slightly increased height
        # DECREASED obstacle_sigma for a smaller, sharper radius
        obstacle_sigma = 20.0

        ys, xs = np.meshgrid(np.arange(height), np.arange(width), indexing='ij')

        rng = np.random.default_rng()

        for _ in range(num_obstacles):
            ox = rng.integers(0.1 * width, 0.9 * width)
            oy = rng.integers(0.1 * height, 0.9 * height)

            # Gaussian bump centered at (ox, oy)
            dist2 = (xs - ox) ** 2 + (ys - oy) ** 2
            bump = obstacle_height * np.exp(-dist2 / (2 * obstacle_sigma ** 2))

            self.P += bump

        # ------------------------------------------------------------------
        # 3. Normalize to [0,1] for visualization
        # ------------------------------------------------------------------
        P_min, P_max = np.min(self.P), np.max(self.P)
        normP = (self.P - P_min) / (P_max - P_min + 1e-12)

        # ------------------------------------------------------------------
        # 4. Create red→blue RGB image
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
        xi = int(np.clip(x, 0, self.P.shape[1] - 2))
        yi = int(np.clip(y, 0, self.P.shape[0] - 1))
        return self.P[yi, xi] - self.P[yi, xi + 1]

    def getGradientY(self, x, y):
        xi = int(np.clip(x, 0, self.P.shape[1] - 1))
        yi = int(np.clip(y, 0, self.P.shape[0] - 2))
        return self.P[yi, xi] - self.P[yi + 1, xi]