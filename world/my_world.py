from swarmy.environment import Environment
import pygame
import numpy as np


class My_environment(Environment):
    def __init__(self, config):
        self.config = config
        super().__init__(config)

        self.definePotentialField()

    def add_static_rectangle_object(self):
        #self.staticRectList.append(['BLACK', pygame.Rect(5, 5, self.config['world_width'] - 10, 5), 5])
        #self.staticRectList.append(['BLACK', pygame.Rect(5, 5, 5, self.config['world_height'] - 10), 5])
        #self.staticRectList.append(
        #    ['BLACK', pygame.Rect(5, self.config['world_height'] - 10, self.config['world_width'] - 10, 5), 5])
        #self.staticRectList.append(
        #    ['BLACK', pygame.Rect(self.config['world_width'] - 10, 5, 5, self.config['world_height'] - 10), 5])
        pass
    def add_static_circle_object(self):
        pass

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
        - a global slope left (high) -> right (low)
        - several Gaussian obstacle hills (local maxima)
        - repulsive potential from top/bottom borders
        """

        width = self.config['world_width']
        height = self.config['world_height']

        # Parameters
        SLOPE_SCALE = 3000.0
        num_obstacles = 12
        obstacle_height = 2000.0
        obstacle_sigma = 20.0
        BORDER_REPULSION_MAGNITUDE = 1000.0
        REPULSION_INFLUENCE_DISTANCE = 50.0

        ys, xs = np.meshgrid(np.arange(height), np.arange(width), indexing='ij')

        # Left to right "slope"
        x_slope = np.linspace(1.0, 0.0, width)
        self.P = np.repeat(x_slope[np.newaxis, :], height, axis=0) * SLOPE_SCALE

        rng = np.random.default_rng()

        for _ in range(num_obstacles):
            ox = rng.integers(0.1 * width, 0.9 * width)
            oy = rng.integers(0.1 * height, 0.9 * height)

            dist2 = (xs - ox) ** 2 + (ys - oy) ** 2
            bump = obstacle_height * np.exp(-dist2 / (2 * obstacle_sigma ** 2))

            self.P += bump

        # Top and bottom repulsion
        dist_to_top = ys
        dist_to_bottom = height - 1 - ys

        dist_min = np.minimum(dist_to_top, dist_to_bottom)

        dist_min_safe = np.clip(dist_min, 1.0, REPULSION_INFLUENCE_DISTANCE)

        repulsion_potential = np.where(
            dist_min < REPULSION_INFLUENCE_DISTANCE,
            BORDER_REPULSION_MAGNITUDE * (1 / dist_min_safe ** 2),
            0.0
        )

        self.P += repulsion_potential

        P_min, P_max = np.min(self.P), np.max(self.P)
        normP = (self.P - P_min) / (P_max - P_min + 1e-12)

        # for color
        R = (255 * normP).astype(np.uint8)
        G = np.zeros_like(R, dtype=np.uint8)
        B = (255 * (1.0 - normP)).astype(np.uint8)

        rgb = np.stack([R, G, B], axis=2)

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