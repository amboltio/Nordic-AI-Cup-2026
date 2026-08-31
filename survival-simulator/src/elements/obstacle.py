import pygame
from typing import Tuple, List

class Obstacle:
    def __init__(self, x: float, y: float, width: float=20, height: float=20, color: Tuple[int, int, int]=(128, 128, 128)):
        self.x: float = x
        self.y: float = y
        self.width: float = width
        self.height: float = height
        self.color: Tuple[int, int, int] = color
        self.edges: List[Tuple[Tuple[float, float], Tuple[float, float]]] = [
            ((self.x, self.y), (self.x + self.width, self.y)),  # top
            ((self.x + self.width, self.y), (self.x + self.width, self.y + self.height)),  # right
            ((self.x + self.width, self.y + self.height), (self.x, self.y + self.height)),  # bottom
            ((self.x, self.y + self.height), (self.x, self.y))  # left
        ]

    def draw(self, screen: pygame.Surface):
        pygame.draw.rect(screen, self.color, (self.x, self.y, self.width, self.height))
    
    def draw_shadow(self, screen: pygame.Surface, color: Tuple[int, int, int]):
        pygame.draw.rect(screen, color, (self.x-5, self.y-5, self.width+6, self.height+6), 5)