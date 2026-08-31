import random
from src.elements.creature import Creature
from src.elements.agent import Agent
from src.elements.fruit import Fruit
from src.elements.tree import Tree
from src.elements.obstacle import Obstacle
from src.elements.predator import Predator
import numpy as np
from collections import defaultdict
from typing import List, Tuple, Optional, Dict, Set
import pygame
from src.elements.biome import Map_generator, smooth_surface


class Environment:
    """
    Environment class defines the environment in which the agents live. Handles all interactions.
    """
    def __init__(self, width: int, height: int, chunk_size: int, rng: random.Random): # Chunk size must at least be the maximum visible distance
        self.rng: random.Random = rng

        self.width: int = width
        self.height: int = height
        self.agents: List[Agent] = []
        self.agents_dict: dict = {} # Dictionary of all agents
        self._next_agent_id: int = 0 # Keep track of agent ids
        self.fruits: List[Fruit] = []
        self.fruits_dict: dict = {}
        self._next_fruit_id: int = 0 # Keep track of fruit ids
        self.trees: List[Tree] = []
        self.obstacles: List[Obstacle] = []
        self.edges: List[Tuple[Tuple[float, float], Tuple[float, float]]] = []
        self.predators: List[Predator] = []
        self.score: float = 0
        self.time: float = 0

        # Set up biomes
        map_generator = Map_generator(self.width, self.height, self.rng, num_biomes=10, num_rivers=1)
        self.biome_map: np.ndarray = map_generator.generate()
        # Cache surface for biome rendering
        self.biome_surface = pygame.Surface((self.width, self.height))
        self._render_biome_surface()
        self.static_surface = self.biome_surface.copy() # Static surface for rendering static elements

        # Cache surfaces for shadow and obstacle rendering
        self.shadow_surface = pygame.Surface((self.width, self.height), pygame.SRCALPHA)
        self.obstacle_surface = pygame.Surface((self.width, self.height), pygame.SRCALPHA)

        # Set up chunking for efficiency
        self.chunk_size: int = chunk_size  # size of each grid cell for spatial partitioning
        self.grid_agents: Dict[Tuple[int, int], Set[Agent]] = defaultdict(set)
        self.grid_fruits: Dict[Tuple[int, int], Set[Fruit]] = defaultdict(set)
        self.grid_trees: Dict[Tuple[int, int], Set[Tree]] = defaultdict(set)
        self.grid_obstacles: Dict[Tuple[int, int], Set[Obstacle]] = defaultdict(set)
        self.grid_edges: Dict[Tuple[int, int], Set[Tuple[Tuple[float, float], Tuple[float, float]]]] = defaultdict(set)
        self.grid_predators: Dict[Tuple[int, int], Set[Predator]] = defaultdict(set)

        # Store all agents observations to avoid recalculating
        self.agent_observations: Dict[int, dict] = {}

        self._create_boundaries(thickness=30) # Encapsulate environment
        self._update_spatial_grid() # Build grid
    
        # Create an offscreen surface the size of the environment
        self.world_surface = pygame.Surface((self.width, self.height), pygame.SRCALPHA)
        # Create vision and leaf surfaces
        self.vision_screen = pygame.Surface((self.width, self.height), pygame.SRCALPHA)
        self.leaf_screen = pygame.Surface((self.width, self.height), pygame.SRCALPHA)


    def _render_biome_surface(self):
        for x in range(self.width):
            for y in range(self.height):
                biome = self.biome_map[x, y]
                color = self.rng.choice(biome.color_palette)
                self.biome_surface.set_at((x, y), color)
        # Optional: smooth edges
        smooth_surface(self.biome_surface, size=3)

    # ------------------- Grid functions -------------------
    def to_chunk(self, x: int, y: int) -> Tuple[int, int]:
        """
        Returns the chunk that entity is in.
        """
        return int(x // self.chunk_size), int(y // self.chunk_size)

    def _update_spatial_grid(self):
        """Rebuilds the spatial grid for all entities."""
        self._update_agent_grid()
        self._update_fruit_grid()
        self._update_tree_grid()
        self._update_obstacle_grid()
        self._update_predator_grid()
        self._update_edge_grid()

    def _update_agent_grid(self):
        """Rebuilds the spatial grid for all agents."""
        self.grid_agents = defaultdict(set)
        for agent in self.agents:
            cx, cy = self.to_chunk(agent.x, agent.y)
            self.grid_agents[(cx, cy)].add(agent)
    
    def _update_fruit_grid(self):
        """Rebuilds the spatial grid for all fruits."""
        self.grid_fruits = defaultdict(set)
        for fruit in self.fruits:
            cx, cy = self.to_chunk(fruit.x, fruit.y)
            self.grid_fruits[(cx, cy)].add(fruit)
    
    def _update_tree_grid(self):
        """Rebuilds the spatial grid for all trees."""
        self.grid_trees = defaultdict(set)
        for tree in self.trees:
            cx, cy = self.to_chunk(tree.x, tree.y)
            self.grid_trees[(cx, cy)].add(tree)
    
    def _update_obstacle_grid(self):
        """Rebuilds the spatial grid for all obstacles."""
        self.grid_obstacles = defaultdict(set)
        for obs in self.obstacles:
            # Obstacle may span multiple chunks
            min_cx, min_cy = self.to_chunk(obs.x, obs.y)
            max_cx, max_cy = self.to_chunk(obs.x + obs.width, obs.y + obs.height)
            for cx in range(min_cx, max_cx + 1):
                for cy in range(min_cy, max_cy + 1):
                    self.grid_obstacles[(cx, cy)].add(obs)
    
    def _update_edge_grid(self):
        """Rebuilds the spatial grid for all edges."""
        self.grid_edges = defaultdict(set)
        for edge in self.edges:
            # Edge may span multiple chunks
            min_cx, min_cy = self.to_chunk(edge[0][0], edge[0][1])
            max_cx, max_cy = self.to_chunk(edge[1][0], edge[1][1])
            for cx in range(min_cx, max_cx + 1):
                for cy in range(min_cy, max_cy + 1):
                    self.grid_edges[(cx, cy)].add(edge)
    
    def _update_predator_grid(self):
        """Rebuilds the spatial grid for all predators."""
        self.grid_predators = defaultdict(set)
        for predator in self.predators:
            cx, cy = self.to_chunk(predator.x, predator.y)
            self.grid_predators[(cx, cy)].add(predator)

    def _get_neighboring_chunks(self, x: int, y: int) -> List[Tuple[int, int]]:
        """Return the 9 chunks surrounding (x,y) including its own."""
        cx, cy = int(x // self.chunk_size), int(y // self.chunk_size)
        chunks: List[Tuple[int, int]] = []
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                chunks.append((cx + dx, cy + dy))
        return chunks

    def _get_local_objects(self, creature: Creature): 
        """Return entities in the creature's neighboring chunks."""
        local_agents = set()
        local_fruits = set()
        local_trees = set()
        local_obstacles = set()
        local_predators = set()
        local_edges = set()
        chunks = self._get_neighboring_chunks(creature.x, creature.y)
        for ch in chunks:
            local_agents.update(self.grid_agents.get(ch, []))
            local_fruits.update(self.grid_fruits.get(ch, []))
            local_trees.update(self.grid_trees.get(ch, []))
            local_obstacles.update(self.grid_obstacles.get(ch, []))
            local_predators.update(self.grid_predators.get(ch, []))
            local_edges.update(self.grid_edges.get(ch, []))

        # Remove self from agents/predators
        if creature in local_agents:
            local_agents.remove(creature)
        elif creature in local_predators:
            local_predators.remove(creature)

        return local_agents, local_fruits, local_trees, local_obstacles, local_predators, local_edges
    
    def _get_local_agents(self, creature: Creature):
        """Return only the agents in neighboring chunks."""
        local_agents = set()
        chunks = self._get_neighboring_chunks(creature.x, creature.y)
        for ch in chunks:
            local_agents.update(self.grid_agents.get(ch, []))
        # Remove self from agents
        if creature in local_agents:
            local_agents.remove(creature)
        return local_agents
    
    def _get_local_predators(self, creature: Creature):
        """Return only the predators in the creature's neighboring chunks."""
        local_predators = set()
        chunks = self._get_neighboring_chunks(creature.x, creature.y)
        for ch in chunks:
            local_predators.update(self.grid_predators.get(ch, []))
        if creature in local_predators:
            local_predators.remove(creature)
        return local_predators
    
    def _get_local_fruits(self, creature: Creature) -> Set[Fruit]:
        """Return only the fruits in the creature's neighboring chunks."""
        local_fruits = set()
        chunks = self._get_neighboring_chunks(creature.x, creature.y)
        for ch in chunks:
            local_fruits.update(self.grid_fruits.get(ch, []))
        return local_fruits
    
    def _get_local_trees(self, creature: Creature):
        """Return only the trees in the creature's neighboring chunks."""
        local_trees = set()
        chunks = self._get_neighboring_chunks(creature.x, creature.y)
        for ch in chunks:
            local_trees.update(self.grid_trees.get(ch, []))
        return local_trees
    
    def _get_local_obstacles(self, creature: Creature):
        """Return only the obstacles in the creature's neighboring chunks."""
        local_obstacles = set()
        chunks = self._get_neighboring_chunks(creature.x, creature.y)
        for ch in chunks:
            local_obstacles.update(self.grid_obstacles.get(ch, []))
        return local_obstacles
    
    def _get_local_edges(self, creature: Creature):
        """Return only the edges in the creature's neighboring chunks."""
        local_edges = set()
        chunks = self._get_neighboring_chunks(creature.x, creature.y)
        for ch in chunks:
            local_edges.update(self.grid_edges.get(ch, []))
        return local_edges



    # ------------------- Spawn/remove functions -------------------

    def _is_position_free(self, x: int, y: int, width: int, height: int) -> bool:
        """
        Check if a position is free. Useful before spawning entities.
        """
        if x < 0 or y < 0 or x + width > self.width or y + height > self.height:
            return False
        
        for obs in self.obstacles:
            if obs.x < x + width and obs.x + obs.width > x and obs.y < y + height and obs.y + obs.height > y:
                return False
        return True

    def spawn_obstacle(self, x: Optional[int]=None, y: Optional[int]=None, width: Optional[int]=None, height: Optional[int]=None, color: Tuple[int, int, int]=(128, 128, 128)) -> Obstacle:
        """
        Spawn an obstacle in the environment.
        """
        if width is None:
            width = self.rng.uniform(30, 100)
        if height is None:
            height = self.rng.uniform(30, 100)
        if x is None:
            x = self.rng.uniform(0, self.width - width)
        if y is None:
            y = self.rng.uniform(0, self.height - height)

        obs = Obstacle(x, y, width=width, height=height, color=color)
        self.obstacles.append(obs)
        shadow_color = np.clip(obs.color - np.array([50,50,50]), 0, 255)
        obs.draw_shadow(self.shadow_surface, color=tuple(shadow_color))
        pygame.draw.rect(self.obstacle_surface, color, pygame.Rect(obs.x, obs.y, obs.width, obs.height)) # Update static surface

        # Rebuild all edges
        self.edges = set()
        for obs in self.obstacles:
            self.edges.update([
                ((obs.x, obs.y), (obs.x + obs.width, obs.y)),  # top
                ((obs.x + obs.width, obs.y), (obs.x + obs.width, obs.y + obs.height)),  # right
                ((obs.x, obs.y + obs.height), (obs.x + obs.width, obs.y + obs.height)),  # bottom
                ((obs.x, obs.y), (obs.x, obs.y + obs.height))  # left
            ])

        self._update_obstacle_grid()
        self._update_edge_grid()
        return obs

    def spawn_agent(self, x: Optional[int]=None, y: Optional[int]=None, parent: Optional[Agent]=None) -> Agent:
        """
        Spawn an agent in the environment.
        """
        if parent is not None: # Child inherits and mutates from parent
            # Spawn near parent
            angle = self.rng.uniform(0, 2 * np.pi)
            dist = self.rng.uniform(10, 30)
            x = parent.x + dist * np.cos(angle)
            y = parent.y + dist * np.sin(angle)
            # Ensure within bounds
            x = min(max(x, 20), self.width - 20)
            y = min(max(y, 20), self.height - 20)

            # Mutate
            mutation_chance = 0.1  # 10% chance per trait to mutate
            mutation_strength = 0.5  # up to ±50%

            # Mutate traits with some probability
            if self.rng.random() < mutation_chance:
                speed = parent.speed * self.rng.uniform(1 - mutation_strength, 1 + mutation_strength)
            else:
                speed = parent.speed

            if self.rng.random() < mutation_chance:
                sprint_speed = parent.sprint_speed * self.rng.uniform(1 - mutation_strength, 1 + mutation_strength)
            else:
                sprint_speed = parent.sprint_speed

            if self.rng.random() < mutation_chance:
                max_energy = parent.max_energy * self.rng.uniform(1 - mutation_strength, 1 + mutation_strength)
            else:
                max_energy = parent.max_energy

            if self.rng.random() < mutation_chance:
                hearing_radius = parent.hearing_radius * self.rng.uniform(1 - mutation_strength, 1 + mutation_strength)
            else:
                hearing_radius = parent.hearing_radius

            if self.rng.random() < mutation_chance:
                vision_radius = parent.vision_radius * self.rng.uniform(1 - mutation_strength, 1 + mutation_strength)
            else:
                vision_radius = parent.vision_radius

            if self.rng.random() < mutation_chance:
                cone_angle = parent.cone_angle * self.rng.uniform(1 - mutation_strength, 1 + mutation_strength)
            else:
                cone_angle = parent.cone_angle


            # Ensure within bounds
            max_speed = 20
            max_sprint_speed = 40
            max_max_energy = 1000
            max_hearing_radius = self.chunk_size/4
            max_vision_radius = self.chunk_size
            max_cone_angle = np.pi/2

            speed = min(speed, max_speed)
            sprint_speed = min(sprint_speed, max_sprint_speed)
            max_energy = min(max_energy, max_max_energy)
            hearing_radius = min(hearing_radius, max_hearing_radius)
            vision_radius = min(vision_radius, max_vision_radius)
            cone_angle = min(cone_angle, max_cone_angle)

            # Update color to match mutation
            agent = Agent(x, y, rng=self.rng) # Initialize agent to get base values

            # baseline trait values
            baseline_speed = (agent.speed + agent.sprint_speed) / 2
            baseline_vision = (agent.hearing_radius + agent.vision_radius + agent.cone_angle) / 3
            baseline_energy = agent.max_energy

            def normalize(val, baseline, vmin, vmax):
                """
                Map val into 0–255, with baseline -> 128
                """
                # how far between vmin and vmax?
                span = vmax - vmin
                if span == 0:
                    return 128
                rel = (val - baseline) / span   # negative if below baseline, positive if above
                return int(np.clip(128 + 127 * rel, 0, 255))

            # Encode traits directly into color channels
            r = normalize((speed + sprint_speed) / 2, baseline_speed, 0, (max_speed + max_sprint_speed) / 2) # red -> speed
            g = normalize((hearing_radius + vision_radius + cone_angle) / 3, baseline_vision, 0, (max_hearing_radius + max_vision_radius + max_cone_angle) / 3) # green -> vision
            b = normalize(max_energy, baseline_energy, 0, max_max_energy) # blue -> energy

            color = (min(max(r, 0), 255), min(max(g, 0), 255), min(max(b, 0), 255))
            if self._is_position_free(x, y, 20, 20):
                agent = Agent(x=x, y=y, rng=self.rng, speed=speed, sprint_speed=sprint_speed, max_energy=max_energy, hearing_radius=hearing_radius, vision_radius=vision_radius, cone_angle=cone_angle, color=color)
            else: # If there is no free space, spawn at parent's position
                agent = Agent(x=parent.x, y=parent.y, rng=self.rng, speed=speed, sprint_speed=sprint_speed, max_energy=max_energy, hearing_radius=hearing_radius, vision_radius=vision_radius, cone_angle=cone_angle, color=color)

        else: # If there is no parent, spawn randomly
            if x is None:
                x = self.rng.uniform(20, self.width-20)
            if y is None:
                y = self.rng.uniform(20, self.height-20)
            if self._is_position_free(x, y, 20, 20):
                agent = Agent(x=x, y=y, rng=self.rng, energy=150) # Starting agents have more energy
            else: # If there is no free space, spawn at a new random position
                self.spawn_agent()
                return
        
        agent.agent_id = self._next_agent_id
        self._next_agent_id += 1
        self.agents.append(agent)
        self.agents_dict[agent.agent_id] = agent
        self._update_agent_grid()
        return agent
    
    def kill_agent(self, agent: Agent):
        """
        Remove an agent from the environment.
        """
        if agent in self.agents:
            self.agents.remove(agent)
            self.agents_dict.pop(agent.agent_id, None)  # removes if exists, does nothing if not
            self._update_agent_grid()
    

    def spawn_fruit(self, x: Optional[int]=None, y: Optional[int]=None, radius: float=5, max_attempts: int=50) -> Optional[Fruit]:
        """
        Spawn a fruit in the environment.
        """
        if x is None:
            x = self.rng.uniform(25, self.width-25)
        if y is None:
            y = self.rng.uniform(25, self.height-25)
        
        for _ in range(max_attempts): # Try to find a free position
            if self._is_position_free(x, y, radius, radius):
                break
            x = self.rng.uniform(0, self.width - radius)
            y = self.rng.uniform(0, self.height - radius)

        if self._is_position_free(x, y, radius, radius):
            fruit = Fruit(x, y, radius=radius)
            fruit.fruit_id = self._next_fruit_id
            self._next_fruit_id += 1
            self.fruits.append(fruit)
            self.fruits_dict[fruit.fruit_id] = fruit
            self._update_fruit_grid()
            return fruit
        else:
            return None
    
    def spawn_fruit_around_tree(self, tree: Tree) -> Optional[Fruit]:
        angle = self.rng.uniform(0, 2 * np.pi)
        dist = self.rng.uniform(tree.radius, tree.radius * 3)
        x = tree.x + dist * np.cos(angle)
        y = tree.y + dist * np.sin(angle)
        return self.spawn_fruit(x=x, y=y, max_attempts=0) # Try only once

    def remove_fruit(self, fruit: Fruit) -> None:
        """
        Remove a fruit from the environment.
        """
        if fruit in self.fruits: # TODO: Check if this line is necessary
            self.fruits.remove(fruit)
            self.fruits_dict.pop(fruit.fruit_id, None)
            self._update_fruit_grid()
    
    def spawn_tree(self, x: Optional[int]=None, y: Optional[int]=None, max_attempts: int=50) -> Tree:
        """
        Spawn a tree in the environment.
        """
        if x is None:
            x = self.rng.uniform(25, self.width-25)
        if y is None:
            y = self.rng.uniform(25, self.height-25)
        
        for _ in range(max_attempts): # Try to find a free position
            if self._is_position_free(x, y, 25, 25):
                break
            x = self.rng.uniform(0, self.width - 25)
            y = self.rng.uniform(0, self.height - 25)

        if self.biome_map[int(x), int(y)].tree_spawn_rate > self.rng.random(): # Check spawn rate
            tree = Tree(x, y)
            self.trees.append(tree)
            self._update_tree_grid()
            return tree
        else:
            return None
    
    def remove_tree(self, tree: Tree) -> None:
        """
        Remove a tree from the environment.
        """
        if tree in self.trees:
            self.trees.remove(tree)
            self._update_tree_grid()

    def spawn_predator(self, x: Optional[int]=None, y: Optional[int]=None, size: float=10, speed: float=11, sprint_speed: float=15, color: Tuple[int, int, int]=(255, 0, 0)) -> Predator:
        """
        Spawn a predator in the environment.
        """
        if x is None:
            x = self.rng.uniform(20, self.width-20)
        if y is None:
            y = self.rng.uniform(20, self.height-20)
        if self._is_position_free(x, y, size, size):
            predator = Predator(x, y, size=size, speed=speed, sprint_speed=sprint_speed, color=color, rng=self.rng)
            self.predators.append(predator)
            self._update_predator_grid()
            return predator



    # ------------------- Helper functions -------------------

    def update_entity_position(self, entity: Creature, distance: float, direction: Optional[float]=None, local_obstacles: Optional[List[Obstacle]]=None):
        """
        Move an entity in the environment. And subtract energy. If the entity is sprinting, the energy cost per distance is higher.
        """
        # Cost per distance
        walking_cost = 0.05
        sprinting_cost = 0.5

        # Speed cannot be negative
        if distance < 0:
            distance = 0

        # Cap distance at maximum sprint speed for creature
        if distance > entity.sprint_speed: 
            distance = entity.sprint_speed
        
        if entity.energy < entity.max_energy / 5 and distance > entity.speed: # If energy is below 20%, limit distance to walking speed
            distance = entity.speed

        if distance <= entity.speed: # Walking
            self.update_entity_energy(entity, distance * walking_cost)
        else: # Sprinting
            self.update_entity_energy(entity, entity.speed * walking_cost + (distance - entity.speed) * sprinting_cost) # Cost per distance increased after walking speed
        
        # Update direction
        if direction is None:
            direction = entity.direction # Move forward if no direction is specified
        else:
            direction = entity.direction + direction

        # Apply biome movement modifier
        x = min(max(int(entity.x), 0), self.width - 1) # Ensure x,y are within bounds
        y = min(max(int(entity.y), 0), self.height - 1)
        biome_movement_modifier = self.biome_map[x, y].move_penalty
        distance *= biome_movement_modifier

        # Previous position
        prev_x = entity.x
        prev_y = entity.y

        # New position
        new_x = prev_x + distance * np.cos(direction)
        new_y = prev_y + distance * np.sin(direction)

        # --- Collision handling by direction adjustment ---
        if self._in_obstacle((new_x, new_y), radius=entity.size, obstacles=local_obstacles):
            # Start with the intended direction and rotate until clear
            angle_step = np.pi / 18  # 10° steps
            max_attempts = int(2 * np.pi / angle_step) 

            for i in range(max_attempts):
                test_angle = direction + angle_step * ((i+1)//2)*(-1)**i # Alternate left and right checks
                test_x = prev_x + distance * np.cos(test_angle)
                test_y = prev_y + distance * np.sin(test_angle)
                if not self._in_obstacle((test_x, test_y), radius=entity.size, obstacles=local_obstacles):
                    entity.x, entity.y = test_x, test_y
                    break


        else:
            entity.x, entity.y = new_x, new_y
        self._keep_agent_in_bounds(entity)

    def update_entity_direction(self, entity: Creature, turn_angle: float):
        """
        Rotate an entity in the environment.
        """
        entity.direction += turn_angle
        self.update_entity_energy(entity, min(np.pi, abs(turn_angle)) / (2 * np.pi)) # Cost per turn capped at 180°
        

    def update_entity_energy(self, entity: Creature, energy_cost: float):
        """
        Update the energy of an entity in the environment.
        """
        entity.energy -= energy_cost


    def get_agent_state(self, agent_id: int) -> Optional[dict]:
        agent = self.agents_dict.get(agent_id)
        if agent is None:
            return None

        # Pull precomputed observations
        observations = self.agent_observations.get(agent_id, [])

        # Clamp x,y for biome lookup
        ix = min(max(int(agent.x), 0), self.width - 1)
        iy = min(max(int(agent.y), 0), self.height - 1)

        return {
            "agent_id": agent.agent_id,
            "observations": observations,
            "energy": agent.energy,
            "biome": self.biome_map[ix, iy].type,
            "age": agent.age,
            "speed": agent.speed,
            "sprint_speed": agent.sprint_speed,
            "hearing_radius": agent.hearing_radius,
            "vision_angle": agent.cone_angle,
            "vision_range": agent.vision_radius,
            "max_energy": agent.max_energy
        }


        
    def agent_step(self, agent_id: int, move_distance: float, move_direction: float, turn_angle: float, spawn_agent: bool=False):
        """
        Apply an action to an agent in the environment.
        """
        agent = self.agents_dict.get(agent_id)
        if agent is None:
            print(f"Agent {agent_id} is dead or not found")
            return  # skip

        local_obstacles = self._get_local_obstacles(agent)

        # Update agent position
        self.update_entity_position(agent, move_distance, move_direction, local_obstacles)
        self._update_agent_grid()

        # Update agent direction
        self.update_entity_direction(agent, turn_angle)

        # Spawn agent
        if spawn_agent and agent.energy > 100:
            self.spawn_agent(parent=agent)
            agent.energy -= 100

    # ------------------- SIMULATION STEP -------------------

    def non_agent_step(self, dt: float):
        """
        Update environment one step

        dt: time step
        """

        # Step all agent interactions
        for agent in self.agents: # make a copy of the list to allow safe removal
            # Update age and energy
            agent.age += dt # age in seconds

            biome_energy_modifier = self.biome_map[min(max(int(agent.x), 0), self.width - 1), min(max(int(agent.y), 0), self.height - 1)].energy_drain_rate # energy drain modifier based on biome
            agent.energy -= dt * biome_energy_modifier # cost one energy per second to be alive

            if agent.energy <= 0: # Agent is dead
                self.kill_agent(agent)
                continue

            if agent.age > agent.max_age:
                agent.energy -= 0.01 * agent.age # When old, lose more energy
            
            local_agents, local_fruits, local_trees, local_obstacles, local_predators, local_edges = self._get_local_objects(agent)
            
            # Get observation
            observation = agent.observe(
                agents=local_agents,
                fruits=local_fruits,
                trees=local_trees,
                obstacles=local_obstacles,
                predators=local_predators,
                edges=local_edges
            )
            self.agent_observations[agent.agent_id] = observation

            # Handle fruit interactions
            if local_fruits:
                local_fruits = list(local_fruits)
                for fruit in reversed(local_fruits):
                    dx = agent.x - fruit.x
                    dy = agent.y - fruit.y
                    distance = np.hypot(dx, dy)
                    if distance < agent.size + fruit.radius:  # touching
                        agent.energy = min(agent.max_energy, agent.energy + fruit.energy) # Eat fruit
                        self.score += fruit.energy / 1000 # Increase score based on fruit energy
                        self.remove_fruit(fruit) # Remove fruit

        # Step all predators
        for predator in self.predators:
            if predator.resting:
                if predator.energy > predator.max_energy * 0.5: # Wake up when energy is high enough
                    predator.resting = False
                else:
                    predator.energy += dt*30 # restore 30 energy per second
                    continue

            local_agents = self._get_local_agents(predator)
            local_edges = self._get_local_edges(predator)
            local_obstacles = self._get_local_obstacles(predator)
            
            # Ignores everything but agents and edges
            observation = predator.observe(
                agents=local_agents,
                fruits=None,
                trees=None,
                obstacles=None,
                edges=local_edges
            )
            
            signals = predator.step(observation) # Get signals from predator based on observation

            # Handle signals
            if 'move' in signals:
                distance = signals['move']
                direction = signals['direction']
                self.update_entity_position(predator, distance, direction, local_obstacles)
                self._update_predator_grid()

            if 'turn' in signals:
                angle = signals['turn']
                self.update_entity_direction(predator, angle)

            # Handle predator interactions with agents
            local_agents = list(self._get_local_agents(predator)) # convert to list for indexing
            if local_agents:
                agent_positions = np.array([[agent.x, agent.y] for agent in local_agents])
                predator_position = np.array([predator.x, predator.y])

                deltas = agent_positions - predator_position # (num_agents, 2)
                distances = np.hypot(deltas[:,0], deltas[:,1]) # The distance between the predator and each agent
                size_sum = predator.size + np.array([agent.size for agent in local_agents]) # The sum of the predator's size and the size of each agent. Used to check if the predator is touching an agent
                touching = np.where(distances < size_sum)[0] # The indices of the agents that the predator is touching

                for index in reversed(touching): # Reverse the loop to avoid modifying the list while iterating
                    agent = local_agents[index]
                    predator.energy = min(predator.max_energy, predator.energy + agent.energy)
                    self.score -= agent.energy / 100 # Penalize agent for being eaten
                    self.kill_agent(agent)

            if predator.energy <= 0: # Go to sleep if energy is 0
                predator.resting = True
                continue

        # Grow fruits
        for fruit in self.fruits:
            if fruit.age > 100: # Fruit rots over time
                self.remove_fruit(fruit)
                continue
            fruit.grow(amount=2 * dt) # Grow by 2 energy per second

        # Spawn new trees
        tree_spawn_chance = 100/max(1,len(self.trees)/2) * dt # Decrease chance with more trees
        decay_factor = 0.5 ** (self.time / 300) # Halve every 5 simulated minutes
        tree_spawn_chance *= decay_factor

        if self.rng.random() < tree_spawn_chance:
            self.spawn_tree() 

        # Tree / fruit logic
        for tree in self.trees:
            tree.grow(amount=1 * dt)
            if tree.age > 50 + (100-50) * (self.rng.random()**(1/2)): # Tree can die after 50 seconds, with a higher probability as it gets older
                self.remove_tree(tree)
                continue
            elif tree.age >= 20:
                if self.rng.random() < dt * self.biome_map[min(max(int(tree.x), 0), self.width - 1), min(max(int(tree.y), 0), self.height - 1)].fruit_spawn_rate: # Only spawn fruit if tree is big enough
                    self.spawn_fruit_around_tree(tree)

        # Update score and time
        self.time += dt
        self.score += dt

        # Spawn predators
        predator_spawn_chance = (1/max(1,len(self.predators))) * dt * self.time * 0.0001 # Increase chance over time, but decrease with more predators
        if predator_spawn_chance > self.rng.random():
            self.spawn_predator()
        



    # ------------------- BOUNDARIES & COLLISIONS -------------------
    def _create_boundaries(self, thickness: float=10):
        """
        Encapsulate environment
        """
        self.spawn_obstacle(x=0, y=0, width=self.width, height=thickness, color=(100,100,100))
        self.spawn_obstacle(x=0, y=self.height-thickness, width=self.width, height=thickness, color=(100,100,100))
        self.spawn_obstacle(x=0, y=0, width=thickness, height=self.height, color=(100,100,100))
        self.spawn_obstacle(x=self.width-thickness, y=0, width=thickness, height=self.height, color=(100,100,100))

    def _keep_agent_in_bounds(self, agent: Agent):
        agent.x = max(agent.size, min(self.width - agent.size, agent.x))
        agent.y = max(agent.size, min(self.height - agent.size, agent.y))

    def _in_obstacle(self, point: Tuple[float, float], radius: float, obstacles: Optional[List[Obstacle]]=None) -> bool:
        if obstacles is None:
            obstacles = self.obstacles
            print("Warning: No local obstacles given to in_obstacle. Using all obstacles.")

        if not obstacles:
            return False

        # Extract obstacle coordinates as NumPy arrays
        xs = np.array([obs.x for obs in obstacles])
        ys = np.array([obs.y for obs in obstacles])
        widths = np.array([obs.width for obs in obstacles])
        heights = np.array([obs.height for obs in obstacles])

        # Compute boundaries with agent size buffer
        lefts   = xs - radius
        rights  = xs + widths + radius
        tops    = ys - radius
        bottoms = ys + heights + radius

        # Vectorized condition: check if agent is within any obstacle box
        inside_x = (lefts < point[0]) & (point[0] < rights)
        inside_y = (tops < point[1]) & (point[1] < bottoms)
        inside_any = np.any(inside_x & inside_y)

        return bool(inside_any)


    # ------------------- DRAW -------------------
    def draw(self, screen: pygame.Surface):
        """
        Visualize the environment scaled to fit the screen (zoomed out if needed).
        """
        # Clear all layers
        self.world_surface.fill((0, 0, 0, 0))
        self.vision_screen.fill((0, 0, 0, 0))
        self.leaf_screen.fill((0, 0, 0, 0))

        # Draw all layers onto the world surface instead of directly on the screen
        self.world_surface.blit(self.static_surface, (0, 0))
        self.world_surface.blit(self.shadow_surface, (0, 0))
        self.world_surface.blit(self.obstacle_surface, (0, 0))


        # Draw fruits
        for fruit in self.fruits:
            fruit.draw(self.world_surface)

        # Draw agents
        for agent in self.agents:
            agent.draw(self.world_surface, self.vision_screen, self.edges)

        # Draw predators
        for predator in self.predators:
            predator.draw(self.world_surface, self.vision_screen, self.edges)

        # Draw tree shadows
        for tree in self.trees:
            shadow_color = np.clip(tree.color - np.array([50, 50, 50]), 0, 255)
            tree.draw_shadow(self.world_surface, color=tuple(shadow_color))
        
        # Draw trees
        for tree in self.trees:
            tree.draw(self.world_surface, self.leaf_screen)
        
        self.world_surface.blit(self.leaf_screen, (0, 0))
        self.world_surface.blit(self.vision_screen, (0, 0))

        # Compute zoom factor to fit the screen
        zoom_x = screen.get_width() / self.width
        zoom_y = screen.get_height() / self.height
        zoom = min(zoom_x, zoom_y)

        # Scale down and blit to the display
        scaled_world = pygame.transform.smoothscale(
            self.world_surface,
            (int(self.width * zoom), int(self.height * zoom))
        )
        screen.blit(scaled_world, (0, 0))
