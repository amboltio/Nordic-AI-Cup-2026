from pydantic import BaseModel
from typing import List, Dict

class ActionRequest(BaseModel):
    """
    Data transfer object to request an action from specific agent
    """
    agent_id: int
    move_distance: float
    move_direction: float # Relative angle in radians
    turn_angle: float # Relative angle in radians
    spawn_agent: bool # Spawn a new agent

class ObservationResponse(BaseModel):
    """
    Data transfer object to receive an observation for specific agent
    """
    agent_id: int
    energy: float
    biome: str
    age: float

    # Agent attributes
    speed: float
    sprint_speed: float
    hearing_radius: float
    vision_angle: float
    vision_range: float
    max_energy: float

    observations: List[Dict] # List of things the agent can see


class StepResponse(BaseModel):
    """
    Data transfer object to receive a step response
    """
    game_status: str
    score: float
    sim_time: float
    n_agents: int
    agent_status: List[ObservationResponse] = []