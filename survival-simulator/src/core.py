import random
from src.utils.simulation import create_environment, step_environment

class SimulationCore:
    """
    Core simulation logic that can be used locally or via API.
    """

    def __init__(self, env_width=1600, env_height=1200, chunk_size=400,
                 starting_agents=5, starting_predators=0, starting_fruits=None,
                 starting_trees=50, seed=None, dt=1/10):
        self.env_width = env_width
        self.env_height = env_height
        self.chunk_size = chunk_size
        self.starting_agents = starting_agents
        self.starting_predators = starting_predators
        self.starting_fruits = starting_fruits or env_width // 50
        self.starting_trees = starting_trees
        self.dt = dt

        if seed is None: # If no seed is provided, generate a random one
            seed = random.randint(0, 2**32 - 1)
        self.seed = seed
        self.rng = random.Random(seed)

        self.env = self._create_env()

    def _create_env(self):
        return create_environment(
            env_width=self.env_width,
            env_height=self.env_height,
            chunk_size=self.chunk_size,
            starting_agents=self.starting_agents,
            starting_predators=self.starting_predators,
            starting_fruits=self.starting_fruits,
            starting_trees=self.starting_trees,
            rng=self.rng
        )

    def step(self, actions):
        """
        Advance the environment by one timestep using the given actions.
        Returns the updated state.
        """
        state = step_environment(self.env, actions, self.dt)
        return state

    def reset(self):
        """
        Reset the simulation to initial state.
        """
        self.rng = random.Random(self.seed)
        self.env = self._create_env()
