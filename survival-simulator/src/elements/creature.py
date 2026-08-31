import pygame
import numpy as np
import random
from src.utils.sensing import compute_visibility
from typing import List, Tuple, Optional, TYPE_CHECKING
from shapely.geometry import Point, Polygon



class Creature:
    """ 
    Base class for all creatures (agents, predators) 

    Args:
        x (float): x position
        y (float): y position
        size (float): size of the creature
        speed (float): maxspeed of the creature walking
        sprint_speed (float): maxspeed of the creature sprinting (costs more energy than walking)
        color (tuple): color of the creature
        energy (float): starting energy of the creature
        max_energy (float): maximum energy of the creature
    """

    def __init__(self, x: float, y: float, size: float=5, speed:float=10, sprint_speed:float=20, color:Tuple[int, int, int]=(128, 128, 128), energy:float=75.0, max_energy:float=500.0, rng:random.Random=None, hearing_radius:float=50, vision_radius:float=200, cone_angle:float=np.pi/3):
        self.id: int = None
        self.x: float = x
        self.y: float = y
        self.size: float = size
        self.speed: float = speed
        self.sprint_speed: float = sprint_speed
        self.color: Tuple[int, int, int] = color
        self.age: float = 0.0
        self.energy: float = energy
        self.max_energy: float = max_energy
        self.rng: random.Random = rng
        self.direction: float = self.rng.uniform(0, 2*np.pi)

        # Perception
        self.hearing_radius: float = hearing_radius
        self.vision_radius: float = vision_radius
        self.cone_angle: float = cone_angle
        self._vision_poly: List[Tuple[float, float]] = None
    
    def move(self, distance: float, direction: Optional[float]=None):
        """Signal the environment to move the creature."""
        return {"move": distance, "direction": direction}
    
    def turn(self, angle: float):
        """Signal the environment to turn the creature."""
        return {"turn": angle}
    
    def clone(self):
        """Signal the environment to spawn a new creature."""
        if self.energy > 100: # Only spawn if we have enough energy
            return {"spawn_agent": True}
        return {}
    
    def update_vision(self, edges: List[Tuple[Tuple[float, float], Tuple[float, float]]]):
        """
        Compute the vision polygon for this timestep.
        
        Return:
            (list of (x, y) points for a polygon, list of edges that obstruct vision)
        """
        edges = list(edges) if edges is not None else []
        self._vision_poly, observed_edges = compute_visibility(
            self.x, self.y, self.direction,
            self.cone_angle, self.vision_radius, edges
        )
        return self._vision_poly, observed_edges
    
    def relative_distance_angle(self, xs: List[float], ys: List[float]) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute relative distance and angle to a list of points.

        Args:
            xs (list): List of x coordinates.
            ys (list): List of y coordinates.

        Returns:
            distances (numpy.ndarray): Array of distances to points.
            angles (numpy.ndarray): Array of angles to same points.
        """
        xs = np.asarray(xs, dtype=float)
        ys = np.asarray(ys, dtype=float)
        dx = xs - self.x
        dy = ys - self.y
        distances = np.hypot(dx, dy)
        angles = np.arctan2(dy, dx) - self.direction
        angles = (angles + np.pi) % (2 * np.pi) - np.pi # wrap to [-pi, pi]
        return distances, angles

    def observe(self, agents=None, fruits=None, trees=None, obstacles=None, edges=None, predators=None):
        """
        Efficiently determine what the agent can perceive (nearby and visible).
        Uses vectorized NumPy operations and precomputed trigonometry.
        """
        observations = [] # List of observations
        cos_dir, sin_dir = np.cos(-self.direction), np.sin(-self.direction) # Precompute cosine and sine of direction for edge computations
        half_cone = self.cone_angle / 2.0 # Precompute half cone angle for process_objects

        # Get edges only if not provided (inefficient)
        if edges is None and obstacles:
            edges = [e for obs in obstacles for e in obs.edges]

        # Compute vision polygon
        vision_poly, hit_edges = self.update_vision(edges)

        def process_objects(obj_list, tag, include_direction=False, include_id = False):
            """
            Args:
                obj_list (list): List of objects to process.
                tag (str): Tag to use for observations.
                include_direction (bool): Whether to include relative direction.

            Iterates over a list of objects and adds visible observations to the observations list.
            
            """
            if not obj_list:
                return
            
            # Vectorize x and y
            xs = np.fromiter((o.x for o in obj_list), float) 
            ys = np.fromiter((o.y for o in obj_list), float)

            # Compute distances and angles to each object
            distances, angles = self.relative_distance_angle(xs, ys)

            # Determine which objects are visible
            nearby_mask = distances <= self.hearing_radius
            visible_mask = (~nearby_mask) & (distances <= self.vision_radius) & (np.abs(angles) <= half_cone)

            # Nearby (hearing/smelling)
            for idx in np.where(nearby_mask)[0]:
                obs = {"type": tag, "distance": float(distances[idx]), "angle": float(angles[idx])}
                if include_direction:
                    rel_dir = ((np.arctan2(self.y - ys[idx], self.x - xs[idx]) - obj_list[idx].direction + np.pi) % (2*np.pi)) - np.pi
                    obs["rel_dir"] = float(rel_dir)
                if include_id:
                    id = obj_list[idx].agent_id
                    obs["id"] = id
                observations.append(obs)

            # Far (vision)
            cand_idxs = np.where(visible_mask)[0]
            if len(cand_idxs) > 0:
                pts = np.column_stack((xs[cand_idxs], ys[cand_idxs]))
                vision_shape = Polygon([(self.x, self.y)] + vision_poly)
                inside = [vision_shape.contains(Point(x, y)) for x, y in pts]
                for idx, flag in zip(cand_idxs, inside):
                    if flag:
                        obs = {"type": tag, "distance": float(distances[idx]), "angle": float(angles[idx])}
                        if include_direction:
                            rel_dir = ((np.arctan2(self.y - ys[idx], self.x - xs[idx]) - obj_list[idx].direction + np.pi) % (2*np.pi)) - np.pi
                            obs["rel_dir"] = float(rel_dir)
                        if include_id:
                            id = obj_list[idx].agent_id
                            obs["id"] = id
                        observations.append(obs)

        # Process each type of object
        if fruits:    
            process_objects(fruits, "Fruit")
        if agents:    
            process_objects([a for a in agents if a is not self], "Agent", include_direction=True, include_id=True)
        if predators: 
            process_objects([p for p in predators if p is not self], "Predator", include_direction=True)
        if trees:     
            process_objects(trees, "Tree")

        # Add visible edges
        for (sx, sy), (ex, ey) in hit_edges:
            dxs, dys, dxe, dye = sx - self.x, sy - self.y, ex - self.x, ey - self.y
            rx_s, ry_s = dxs * cos_dir - dys * sin_dir, dxs * sin_dir + dys * cos_dir
            rx_e, ry_e = dxe * cos_dir - dye * sin_dir, dxe * sin_dir + dye * cos_dir
            observations.append({"type": "Edge", "coords": ((rx_s, ry_s), (rx_e, ry_e))}) # coord dimensions: ((start_x, start_y), (end_x, end_y))

        return observations


    def draw(self, screen: pygame.Surface, vision_screen: pygame.Surface, edges: List[Tuple[Tuple[float, float], Tuple[float, float]]]):

        s = vision_screen # Overlay surface to handle vision and hearing visualization

        # Hearing/smelling circle
        near_color = (*self.color, 80)
        pygame.draw.circle(s, near_color, (int(self.x), int(self.y)), self.hearing_radius)

        # Vision polygon
        cone_color = (*self.color, 80)
        if self._vision_poly is None:
            self.update_vision(edges)
        vis_poly = self._vision_poly

        if len(vis_poly) >= 2:
            pygame.draw.polygon(s, cone_color, [(self.x, self.y)] + vis_poly)

        # Draw creature
        pygame.draw.circle(screen, self.color, (int(self.x), int(self.y)), self.size)
        pygame.draw.circle(screen, (0, 0, 0), (int(self.x), int(self.y)), self.size, 1) # outline

        # Draw energy bar
        energy_bar_length = self.size * 2
        energy_bar_height = 4
        bar_x = self.x - self.size
        bar_y = self.y - self.size - 10
        pygame.draw.rect(screen, (20, 20, 20), (bar_x, bar_y, energy_bar_length, energy_bar_height))  # energy bar background
        pygame.draw.rect(screen, (0, 255, 0), (bar_x, bar_y, (self.energy / self.max_energy) * energy_bar_length, energy_bar_height))  # current energy

