import random
from fastapi import FastAPI, Body
from src.utils.DTOs import StepResponse
from src.utils.controllers.dummy_agent_policy import action_decision

HOST = "0.0.0.0"
PORT = 9052

app = FastAPI(title="Survival Simulator Agent Endpoint")

@app.post("/predict")
def predict(step: StepResponse = Body(...)):
    """
    Receives the current simulation state and returns actions for all agents.
    """
    rng = random.Random(1)  # deterministic for testing
    actions = [action_decision(agent.dict(), rng).dict() for agent in step.agent_status]
    
    # Must return {"actions": [...]} format
    return {"actions": actions}

@app.get("/")
def index():
    return {"message": "Agent endpoint running!"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host=HOST, port=PORT)