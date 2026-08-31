import random
import numpy as np
from src.utils.DTOs import ActionRequest

def action_decision(observation_response: dict, rng: random.Random):
    """
    Dummy action selection for the agent
    
    Args:
        observation_response (dict): Observation response from the environment
        rng (random.Random): Random number generator

    Returns:
        ActionRequest: Action decision
    """
    # Unpack DTO
    agent_id = observation_response["agent_id"]
    observations = observation_response["observations"]
    energy = observation_response["energy"]
    biome = observation_response["biome"]
    age = observation_response["age"]
    speed = observation_response["speed"]
    sprint_speed = observation_response["sprint_speed"]
    hearing_radius = observation_response["hearing_radius"]
    vision_angle = observation_response["vision_angle"]
    vision_range = observation_response["vision_range"]
    max_energy = observation_response["max_energy"]

    move_distance = rng.uniform(0.0, sprint_speed) # Move random distance
    move_direction = 0.0 # Move forward
    turn_angle = rng.uniform(-np.pi/4, np.pi/4) # Turn random angle up to 45 degrees
    spawn_agent = True # Spawn new agent if energy is high enough


    return ActionRequest(
        agent_id=agent_id,
        move_distance=move_distance,
        move_direction=move_direction,
        turn_angle=turn_angle,
        spawn_agent=spawn_agent,
    )
