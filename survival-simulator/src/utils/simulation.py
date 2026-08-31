from src.elements.environment import Environment


def create_environment(env_width, env_height, chunk_size, starting_agents,
                       starting_predators, starting_fruits, starting_trees, rng):
    """
    Initialize an environment for simulation.
    
    Args:
        env_size (int): Width and height of the environment in pixels.
        chunk_size (int): Spatial grid chunk size.
        starting_agents (int): Initial number of agents.
        starting_predators (int): Initial number of predators.
        starting_fruits (int): Initial number of fruits.
        rng (random.Random): Random number generator.

    Returns:
        Environment: The initialized environment object.
    """
    env = Environment(env_width, env_height, chunk_size, rng)

    obstacles = env_width // 20

    for _ in range(obstacles):
        env.spawn_obstacle()
    for _ in range(starting_agents):
        env.spawn_agent()
    for _ in range(starting_predators):
        env.spawn_predator()
    for _ in range(starting_fruits):
        env.spawn_fruit()
    for _ in range(starting_trees):
        env.spawn_tree()
    for tree in env.trees:
        tree.grow(amount=rng.uniform(20, 80)) # Random initial size

    # Maintain agent dict for safe access
    env.agents_dict = {agent.agent_id: agent for agent in env.agents}

    return env


def step_environment(env, actions, dt=1/10):
    """
    Apply agent actions and advance the environment by one timestep.

    Args:
        env (Environment): The simulation environment.
        actions (list): List of (agent_id, action) pairs.
        dt (float): Delta time per step.

    Returns:
        dict: A dictionary containing updated state (score, alive agents, observations).
    """
    # Step each agent
    for agent_id, action in actions:
        if agent_id not in env.agents_dict: # Skip if agent has died
            continue
        env.agent_step(
            agent_id,
            move_distance=action.move_distance,
            move_direction=action.move_direction,
            turn_angle=action.turn_angle,
            spawn_agent=action.spawn_agent
        )

    # Step non-agent entities (fruits, predators, etc.)
    env.non_agent_step(dt)

    # Update agent dict
    env.agents_dict = {agent.agent_id: agent for agent in env.agents}
    

    return {
        "score": env.score,
        "sim_time": env.time,
        "num_agents": len(env.agents),
        "observations": [env.get_agent_state(agent.agent_id) for agent in env.agents]
    }
