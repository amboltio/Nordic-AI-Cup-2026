import random
import numpy as np
import pygame
import scipy.ndimage
from scipy.ndimage import distance_transform_edt
from typing import List, Type, Tuple, Optional
import cProfile
import pstats
from pstats import SortKey
import time

class Biome:
    def __init__(self, type: str) -> None:
        self.type: str = type # e.g., "forest", "desert", "swamp"
        self.color_palette: List[Tuple[int, int, int]] = [] # List of colors representing the biome
        self.move_penalty: float = 1.0 # Multiplier for movement speed in given biome
        self.tree_spawn_rate: float = 1.0 # Odds of spawning a tree here
        self.fruit_spawn_rate: float = 0.0 # fruits per second per 100x100 area
        self.energy_drain_rate: float = 1.0 # Multiplier for energy drain in given biome


class Forest_biome(Biome):
    def __init__(self):
        super().__init__("forest")
        self.tree_spawn_rate = 1.0
        self.fruit_spawn_rate = 0.1 # fruits per second per 100x100 area
        self.color_palette = [(0, 100, 0), (20, 107, 13), (50, 69, 19)] # greens and browns

class Grassland_biome(Biome):
    def __init__(self):
        super().__init__("grassland")
        self.tree_spawn_rate = 0.5
        self.fruit_spawn_rate = 0.1
        self.color_palette = [(34, 139, 34), (30, 144, 50), (50, 205, 64)] # greens and browns


class Swamp_biome(Biome):
    def __init__(self):
        super().__init__("swamp")
        self.tree_spawn_rate = 0.9
        self.fruit_spawn_rate = 0.08
        self.color_palette = [(47, 79, 79), (0, 100, 0), (85, 107, 47), (139, 69, 19)] # dark greens and browns
        self.move_penalty = 0.5


class Desert_biome(Biome):
    def __init__(self):
        super().__init__("desert")
        self.move_penalty = 0.8
        self.tree_spawn_rate = 0.1
        self.fruit_spawn_rate = 0.05
        self.color_palette = [(210, 180, 140), (244, 164, 96), (222, 184, 135), (205, 133, 63)] # tans and browns

class River_biome(Biome):
    def __init__(self):
        super().__init__("river")
        self.move_penalty = 0.3
        self.tree_spawn_rate = 0.0
        self.fruit_spawn_rate = 0.0
        self.color_palette = [(85, 206, 255), (90, 200, 255), (80, 190, 250), (85, 200, 255)] # blues
        self.stream_flow_speed: float = 5.0 # Speed at which entities are pushed along the river


class Map_generator:
    def __init__(self, width: int, height: int, rng: random.Random, num_biomes: int=4, num_rivers: int=1):
        self.width: int = width
        self.height: int = height
        self.num_biomes: int = num_biomes
        self.rng: random.Random = rng
        self.biome_types: List[Type[Biome]] = [Forest_biome, Swamp_biome, Desert_biome, Grassland_biome]
        self.num_rivers: int = num_rivers
    
    def generate(self, points: Optional[List[Tuple[int, int]]] = None):
        if points is None:
            points = [(self.rng.randint(0, self.width-1), self.rng.randint(0, self.height-1)) for _ in range(self.num_biomes)]
        biomes: List[Biome] = [self.rng.choice(self.biome_types)() for _ in range(self.num_biomes)]

        # Convert to arrays
        px = np.array([p[0] for p in points])[:, None, None]  # (N,1,1)
        py = np.array([p[1] for p in points])[:, None, None]  # (N,1,1)

        # Grid of coordinates
        gx, gy = np.meshgrid(np.arange(self.width), np.arange(self.height), indexing="ij")

        # Compute squared distances for all seeds at once
        dists = (gx[None, :, :] - px) ** 2 + (gy[None, :, :] - py) ** 2  # (N, W, H)
        # Find nearest biome index per pixel
        nearest = np.argmin(dists, axis=0)  # (W, H)

        # Fill biome map
        biome_map: np.ndarray = np.empty((self.width, self.height), dtype=object)
        for i, biome in enumerate(biomes):
            biome_map[nearest == i] = biome
        
        # Add rivers
        for _ in range(self.num_rivers):
            river_start = self.rng.choice(['top', 'bottom', 'left', 'right'])
            river_end = self.rng.choice(['top', 'bottom', 'left', 'right'])

            start_point = self._get_edge_point(river_start)
            end_point = self._get_edge_point(river_end)
            
            # Generate river path
            river_path = self._generate_river_path(start_point, end_point)

            # Optimized river widening using distance transform
            radius = self.rng.randint(20, 100)
            river_mask = self._create_river_mask(river_path, radius)

            # Apply mask to biome_map
            biome_map[river_mask] = River_biome()

        return biome_map
    
    def _get_edge_point(self, edge: str) -> Tuple[int, int]:
        """Get a random point on the specified edge."""
        if edge == 'top':
            return (self.rng.randint(0, self.width-1), 0)
        elif edge == 'bottom':
            return (self.rng.randint(0, self.width-1), self.height - 1)
        elif edge == 'left':
            return (0, self.rng.randint(0, self.height-1))
        else:  # right
            return (self.width - 1, self.rng.randint(0, self.height-1))
    
    def _generate_river_path(self, start_point: Tuple[int, int], end_point: Tuple[int, int]) -> List[Tuple[int, int]]:
        """Generate river path using pseudo random walk."""
        river_path = []
        current_point = start_point
        max_river_turn = np.radians(10)
        direction = np.arctan2(end_point[1] - start_point[1], end_point[0] - start_point[0])

        safety_counter = 0
        while np.hypot(current_point[0] - end_point[0], current_point[1] - end_point[1]) > 5 and safety_counter < 10000: # prevent infinite loops
            river_path.append(current_point)
            direction += self.rng.uniform(-max_river_turn, max_river_turn)
            
            if self.rng.random() < 0.9: # 90% chance to adjust towards end point
                target_direction = np.arctan2(end_point[1] - current_point[1], end_point[0] - current_point[0])
                direction += 0.3 * (target_direction - direction + np.pi) / (2 * np.pi) - np.pi

            step_size = self.rng.randint(3, 7)
            new_x = int(current_point[0] + step_size * np.cos(direction))
            new_y = int(current_point[1] + step_size * np.sin(direction))
            new_x = max(0, min(self.width - 1, new_x))
            new_y = max(0, min(self.height - 1, new_y))

            if new_x == current_point[0] and new_y == current_point[1]: # if stuck, move directly towards end point
                dx = 1 if end_point[0] > current_point[0] else -1 if end_point[0] < current_point[0] else 0
                dy = 1 if end_point[1] > current_point[1] else -1 if end_point[1] < current_point[1] else 0
                new_x = max(0, min(self.width - 1, current_point[0] + dx))
                new_y = max(0, min(self.height - 1, current_point[1] + dy))

            current_point = (new_x, new_y)
            safety_counter += 1
        
        #river_path.append(end_point)
        return river_path
    
    def _create_river_mask(self, river_path: List[Tuple[int, int]], radius: int) -> np.ndarray:
        """
        Create river mask using distance transform.
        
        1. Mark the river path pixels
        2. Compute distance from each pixel to nearest river pixel
        3. Include all pixels within radius distance
        """
        river_mask = np.zeros((self.width, self.height), dtype=bool) # Initialize mask
        
        # Mark river path pixels
        for rx, ry in river_path:
            if 0 <= rx < self.width and 0 <= ry < self.height:
                river_mask[rx, ry] = True
        
        # Use distance transform
        distances = distance_transform_edt(~river_mask) # distances to nearest river pixel
        river_mask = distances <= radius
        
        return river_mask

    def render(self, screen, biome_map: np.ndarray):
        """Render the biome map to the screen."""
        arr = np.zeros((self.width, self.height, 3), dtype=np.uint8)
        for x in range(self.width):
            for y in range(self.height):
                biome = biome_map[x, y]
                arr[x, y] = self.rng.choice(biome.color_palette)
        pygame.surfarray.blit_array(screen, arr)

def smooth_surface(screen, size=3):
    """Apply a box blur to smooth sharp biome edges."""
    arr = pygame.surfarray.array3d(screen)
    smoothed = scipy.ndimage.uniform_filter(arr, size=(size, size, 1))
    pygame.surfarray.blit_array(screen, smoothed.astype(np.uint8))


# Test
if __name__ == "__main__":

    seed = random.randint(0, 100000)
    WIDTH, HEIGHT = 1600, 1200
    rng = random.Random(seed)

    profiler = cProfile.Profile()
    profiler.enable()

    generator = Map_generator(WIDTH, HEIGHT, rng, num_biomes=20, num_rivers=1)

    start_time = time.time()
    biome_map = generator.generate()
    end_time = time.time()

    profiler.disable()
    stats = pstats.Stats(profiler)
    stats.sort_stats(SortKey.CUMULATIVE)
    stats.print_stats(20)

    print(f"Map generation took {end_time - start_time:.2f} seconds. With seed {seed}.")

    print(f"Using seed: {seed}")

    pygame.init()
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption("Voronoi Biome Map")
    generator.render(screen, biome_map)
    smooth_surface(screen, size=3)
    pygame.display.flip()
    
    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
    pygame.quit()