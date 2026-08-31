import numpy as np
from src.elements.creature import Creature
from typing import Tuple, List, Optional
import random

class Predator(Creature):
    """
    Predator class that will chase the closest agent
    """
    def __init__(self, x: float, y: float, size: float=10, speed: float=11, sprint_speed: float=15, color:Tuple[int, int, int]=(255, 0, 0), energy: float=0.0, max_energy: float=200.0, rng: random.Random=None):
        super().__init__(x, y, size, speed, sprint_speed, color, energy, max_energy, rng=rng)
        self.resting: bool = True
        self.hearing_radius: float = 60
        self.vision_radius: float = 250

    def step(self, observation: Optional[List[dict]]=None) -> dict:
        """
        observation: list of visible entities eg.
          ('Agent', distance, angle),
          ('Edge', ((rx_s, ry_s), (rx_e, ry_e))),
        where Edge coords are already in the predator-local frame (origin at predator, rotated).
        """
        signals: dict = {}


        # Only consider agents and edges
        agents = [o for o in observation if o.get("type") == "Agent"]
        edges = [o["coords"] for o in observation if o.get("type") == "Edge"]

        # Chase closest agent
        if agents:
            closest_agent = min(agents, key=lambda f: f["distance"])
            distance_to_agent = closest_agent["distance"]
            angle_to_agent = closest_agent["angle"]
            agent_looking_dir = closest_agent["rel_dir"]

            if abs(agent_looking_dir) > np.pi*1/2 or distance_to_agent < self.hearing_radius * 1.5: # Only chase when behind
                turn_strength = max(-0.3, min(0.3, angle_to_agent * 0.5))
                if abs(angle_to_agent) > 0.05:
                    signals.update(self.turn(turn_strength))
                    signals.update(self.move(min(self.sprint_speed, distance_to_agent), turn_strength))
                else: # within 5 degrees, just move forward to avoid jitter
                    signals.update(self.move(min(self.sprint_speed, distance_to_agent), angle_to_agent))
                return signals
            
            else: # Agent looking towards predator, pivot around it
                # Move around agent to avoid agent vision cone and turn towards it afterwards
                pivot_sign = -np.sign(agent_looking_dir) # Opposite of agent looking direction
                move_dir = angle_to_agent +pivot_sign * np.pi*1/4 # Move at an angle relative to agent
                signals.update(self.move(self.sprint_speed, move_dir))
                
                # Compute predicted new angle to agent after moving
                dx_move = self.sprint_speed * np.cos(move_dir)
                dy_move = self.sprint_speed * np.sin(move_dir)
                x_agent = distance_to_agent * np.cos(angle_to_agent)
                y_agent = distance_to_agent * np.sin(angle_to_agent)
                x_new = x_agent - dx_move
                y_new = y_agent - dy_move
                new_angle_to_agent = np.arctan2(y_new, x_new)

                signals.update(self.turn(new_angle_to_agent))
                return signals

        # Avoid edges
        if edges:

            def closest_point_on_edge_to_origin(edge):
                """
                Returns the closest point on the edge to the predator
                """
                (x1, y1), (x2, y2) = edge
                dx, dy = x2 - x1, y2 - y1 # w = (x2-x1, y2-y1)
                t = (-(x1 * dx + y1 * dy)) / (dx*dx + dy*dy) # t = (w dotproduct v) / v²
                t = max(0.0, min(1.0, t)) # Check if t is between 0 and 1, otherwise the closest point is the edge end
                return (x1 + t * dx, y1 + t * dy)
            
            # find closest point on each edge relative to origin
            pts = [closest_point_on_edge_to_origin(e) for e in edges]

            # pick closest from those
            closest = min(pts, key=lambda p: np.hypot(p[0], p[1]))
            dist_to_closest = max(np.hypot(closest[0], closest[1]) - self.size, 2.0)

            angle_to_edge = np.arctan2(closest[1], closest[0])
            angle_to_edge = (angle_to_edge + np.pi) % (2 * np.pi) - np.pi

            if angle_to_edge > 0:
                turn_angle = -np.pi / dist_to_closest
            else:
                turn_angle = np.pi / dist_to_closest

            signals.update(self.turn(turn_angle))
            signals.update(self.move(self.speed, turn_angle))

            return signals

        # Default wandering
        signals.update(self.turn(self.rng.uniform(-0.1, 0.1)))
        signals.update(self.move(self.speed, None))
        return signals
