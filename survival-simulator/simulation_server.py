import requests
import time
from src.core import SimulationCore
from src.utils.DTOs import StepResponse, ObservationResponse, ActionRequest

def run_local_game(agent_url: str, seed=None):
    # Initialize simulation
    print("Initializing simulation...")
    sim = SimulationCore(seed=seed)
    print("Simulation initialized.")
    step_response = StepResponse(game_status="ok", score=0, sim_time=sim.env.time, n_agents=len(sim.env.agents), agent_status=[]) # Run first step without actions to get initial observations

    while step_response.game_status != "game_over":
        # Ask agent server for actions
        try:
            resp = requests.post(agent_url, json=step_response.dict(), timeout=10)
            resp.raise_for_status()
            actions = [ActionRequest(**a) for a in resp.json().get("actions", [])]
        except Exception as e:
            print(f"Error contacting agent: {e}")
            break

        # Step simulation
        parsed_actions = [(a.agent_id, a) for a in actions]
        state = sim.step(parsed_actions)

        # Build StepResponse for next step
        status = [
            ObservationResponse(
                agent_id=obs["agent_id"],
                observations=obs["observations"],
                energy=obs["energy"],
                biome=obs["biome"],
                age=obs["age"],
                speed=obs["speed"],
                sprint_speed=obs["sprint_speed"],
                hearing_radius=obs["hearing_radius"],
                vision_angle=obs["vision_angle"],
                vision_range=obs["vision_range"],
                max_energy=obs["max_energy"]
            )
            for obs in state["observations"] if obs is not None
        ]

        step_response = StepResponse(
            game_status="ok" if state["num_agents"] > 0 and sim.env.time <= 3000 else "game_over",
            score=state["score"],
            sim_time=state["sim_time"],
            n_agents=state["num_agents"],
            agent_status=status
        )

        print(f"Score: {step_response.score:.2f} | Alive: {len(status)} | Time: {sim.env.time:.2f}")

    print(f"Game finished! Final score: {step_response.score}")

if __name__ == "__main__":
    print("Waiting for agent server to be available...")
    AGENT_TEST_URL = "http://127.0.0.1:9052/"

    # Keep trying until the agent responds
    while True:
        try:
            response = requests.get(AGENT_TEST_URL, timeout=1)
            if response.status_code == 200:
                print("Agent connected successfully!")
                break
        except requests.ConnectionError:
            pass  # Agent not up yet
        except requests.Timeout:
            pass  # Just retry infinitely

        time.sleep(0.5)  # Prevent CPU spinning
    AGENT_URL = "http://127.0.0.1:9052/predict"
    run_local_game(AGENT_URL, seed=1)
