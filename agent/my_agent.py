from swarmy.agent import Agent
import random
import pygame
import numpy as np


class MyAgent(Agent):
    def __init__(self, environment, controller, sensor, config):
        super().__init__(environment, controller, sensor, config)

        self.environment = environment
        # Initialize an empty list to store (x, y) points
        self.trajectory = []

    def initial_position(self):
        """
        The controller handles the initial position setting to ensure it starts on the high-potential side.
        """
        # The controller will call set_position, so we just pass here.
        pass

    def save_information(self, last_robot):
        """
        Draw the trajectory of the robot onto the environment surface.
        """
        print(f"Saving information for Agent {self.unique_id}. Trajectory length: {len(self.trajectory)}")

        # Only draw if there are enough points
        if len(self.trajectory) > 1:
            trajectory_color = (0, 0, 0)  # Black color for the trajectory

            # The trajectory is a list of (x, y) tuples
            # We use 1 as the line width
            pygame.draw.lines(
                self.environment.displaySurface,
                trajectory_color,
                False,  # not closed
                self.trajectory,
                1
            )

        # Optionally save the image
        filename = f"agent_{self.unique_id}_trajectory.png"
        pygame.image.save(self.environment.displaySurface, filename)
        print(f"Saved environment image to {filename}")

        pass