import pygame
from typing import Tuple, Optional

class Fruit:
    def __init__(self, x: float, y: float, radius: float=1):
        self.fruit_id: Optional[int] = None
        self.x: float = x
        self.y: float = y
        self.radius: float = radius
        self.color: Tuple[int, int, int] = (0, 255, 0)
        self.energy: float = 20
        self.age: float = 0

    def grow(self, amount: float=1):
        """
        Grow the fruit and increase its energy
        """
        self.age += amount
        if self.energy < 60:
            self.energy += amount
            self.radius +=  amount * 0.1
        else:
            r = min(150, self.color[0] + amount*255/60) # Fruit turns red as it ripens
            g = max(100, self.color[1] - amount*255/60) # Fruit turns brown as it over-ripens
            b = 0
            self.color = (r, g, b)

    def draw(self, screen: pygame.Surface):
        pygame.draw.circle(screen, self.color, (int(self.x), int(self.y)), self.radius)