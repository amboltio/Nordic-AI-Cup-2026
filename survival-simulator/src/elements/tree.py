import pygame
from typing import Tuple

class Tree:
    def __init__(self, x: float, y: float, radius: float=10, color: Tuple[int, int, int]=(41*2, 11*2, 6*2)):
        self.x = x
        self.y = y
        self.radius = radius
        self.color = color
        self.age = 0

    def grow(self, amount: float=1):
        self.radius = min(20, self.radius + amount)
        self.age += amount
    
    def draw(self, screen: pygame.Surface, leaf_screen: pygame.Surface):
        # Draw tree
        pygame.draw.circle(screen, self.color, (int(self.x), int(self.y)), self.radius)
        
        # Draw leaves
        leaf_color = (34, 139, 34, min(150, 50 + self.age * 10)) # More opaque as tree ages
        pygame.draw.circle(leaf_screen, leaf_color, (int(self.x), int(self.y)), self.radius + 10 + self.age // 2)

                
    def draw_shadow(self, screen: pygame.Surface, color: Tuple[int, int, int]):
        pygame.draw.circle(screen, color, (int(self.x-3), int(self.y-3)), self.radius+1)