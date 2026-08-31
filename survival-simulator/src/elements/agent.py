from src.elements.creature import Creature
from typing import Tuple, Optional
import random
import numpy as np

class Agent(Creature):
    """
    Agent class representing the players characters in the game
    """
    def __init__(self,
                x: float, 
                y: float, 
                size: float=5, 
                speed: float=10, 
                sprint_speed: float=20, 
                energy: float=75.0, 
                max_energy: float=500.0, 
                rng: Optional[random.Random]=None, 
                max_age: Optional[float] = None,
                hearing_radius:float=50,
                vision_radius:float=200, 
                cone_angle:float=np.pi/3,
                color: Tuple[int, int, int]=(128, 128, 128) # Base color before mutations
                ):
        
        super().__init__(x=x, y=y, size=size, speed=speed, sprint_speed=sprint_speed, color=color, energy=energy, max_energy=max_energy, rng=rng, hearing_radius=hearing_radius, vision_radius=vision_radius, cone_angle=cone_angle)
        self.max_age = max_age if max_age is not None else 60 + self.rng.uniform(0, 60) # Agents above max_age will die faster