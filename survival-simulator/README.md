# Survival simulator

Improvise, adapt, overcome!

You are the hivemind of an entire species of herbivores. Ensure their survival by eating fruits and conserving energy, but beware of the predators roaming the territory.

## About the game
You are in control of all members of your species (agents) simultaneously. At every tick you will recieve a list of each agent's observations and status. The agents have a hearing/smelling radius and a vision cone. Any object within the vision cone is added to observations with an object type and data depending on object type. Note that creatures cannot hear/smell walls, only see them and vision can be blocked by walls.

![Entities](/images/Entity_overview.png)

After receiving an observation for each agent you will have to respond with the action for each agent. This includes moving, turning and even spawning new agents.
Agents can move in any direction but are limited by their speed/sprint_speed with sprinting having a higher energy cost per unit traveled. They can also turn at a low energy cost and even spawn new agents at a very high cost.

A newly spawned agent will inherit the parent's traits with a small chance of mutations happening in each trait. These mutations can affect the agent both positively and negatively.

During the simulation predators will spawn in to hunt the agents.

The simulation will run until all agents have died or the environment has simulated 3000 seconds corresponding to 30000 ticks.


## Environment
Running the simulation creates a random environment with different biomes which affect fruit spawn rates and creature movement. The environment will also have a number of obstacles, obstructing vision and movement.
5 agents will spawn in at the start of the simulation as well as some fruits and fruit trees.
During the simulation fruit will spawn around trees as a source of energy.
Both trees and fruit have a life cycle growing/rotting over time. 

Predators will spawn in during the simulation with increased odds over time. The predators will hunt down agents, killing any agent they touch and stealing their remaining energy. This will decrease your score based on the agent's remaining energy!


## Your goal
Your main goal is to keep your species alive for as long as possible.
If your species can consistently survive the entire simulation time, your score can be increased further by eating fruit and avoiding getting eaten by predators.

## Status and Observations

At every environment step you will receive game_status, score and a list of agent_status objects.
The game_status will be "ok" as long as the simulation is running.
The score is the current accumulated score since simulation start.
The content of each entry in the agent_status list can be seen in the table below:
| Name              | Explanation                                                   |
|-------------------|---------------------------------------------------------------|
| agent_id          | ID to keep track of agents                                    |
| observations      | List of observations for the agent                            |
| energy            | Agent's remaining energy                                      |
| biome             | The biome type that the agent is currently in                 |
| age               | How many simulated seconds the agent has been alive           |
| speed             | The maximum speed the agent can move at no additional cost    |
| sprint_speed      | The maximum speed the agent can move (higher energy cost)     |
| hearing_radius    | How far the agent can hear/smell entities                     |
| vision_angle      | The angle of the vision cone (radians)                        |
| vision_range      | How far the agent can see                                     |
| max_energy        | How much energy the agent can store                           |

The speed, sprint_speed, hearing_radius, vision_angle, vision_range, and max_energy describes static agent traits/attributes that can mutate when spawning new agents.

The sense traits/attributes are shown in the following figure:
![Traits](/images/agent_trait_ref.png)

The observations have the following format based on what is being observed:

| Observation type  | Data                                                                  |
|-------------------|-----------------------------------------------------------------------|
| Fruit             | Type, Distance, Angle (radians)                                       |
| Agent             | Type, Distance, Angle (radians), Relative looking direction           |
| Predator          | Type, Distance, Angle (radians), Relative looking direction           |
| Tree              | Type, Distance, Angle (radians)                                       |
| Edge              | Type, Coordinates (start, end)                                        |


Edges are only observed if within the vision cone. Other observations are also observed in the hearing/smell range:
![Sensing](/images/Agent_senses.png)

## Controls
After receiving the list of agent statuses and observations from the environment, your controller must decide what each agent should do during the next simulation step.
This decision should be returned as a list of ActionRequests, one for each agent (See [DTOs.py](src/utils/DTOs.py)).

Each ActionRequest must include the following fields:

|Field	            | Type	| Description                                                               |
|-------------------|-------|---------------------------------------------------------------------------|
|agent_id	        | int	| The ID of the agent this action applies to.                                   |
|move_distance	    | float | Distance to move (capped by speed or sprint_speed).                       |
|move_direction     | float | Absolute movement direction (radians).                                    |
|turn_angle	        | float | Rotation applied this step (radians).                                     |
|spawn_agent	    | bool  | Whether the agent should attempt to spawn a new agent (high energy cost).   |

For a full example of how actions are used in practice, see [dummy_agent_policy.py](src/utils/controllers/dummy_agent_policy.py) and [agent_server.py](agent_server.py).

## Energy costs
|Action                                               | Energy cost                                     |
|-----------------------------------------------------|-------------------------------------------------|
| Walking (move_distance <= speed                     | move_distance * 0.05                            |
| Sprinting (speed <= move_distance <= sprint_speed)  | speed * 0.05 + (move_distance - speed) * 0.5    |
| Turning                                             | abs(turn_angle) / 2 * pi                        |
| Spawning                                            | 100                                             |
| Living (passive cost over time)                     | 1 / 10 * biome_energy_modifier                  |

When agents are older than a randomly chosen age between 60 and 120, their living cost will increase by 0.01 * agent.age.


## Scoring
Your score is mainly determined by how long your species survive. However, eating a fruit will increase your score by a small amount and getting eaten by predators will decrease your score based on how much energy the agent had remaining.

## Validation and Evaluation
To test your model and server connection, start a validation attempt. You can only have one attempt going at once, but attempts are unlimited. Your attempt will be put into a queue, and run when it's your turn. The validation attempts will use random seeds. 

Once you are ready to evaluate your final model, start your evaluation attempt. You only have **ONE** try, so make sure the model is ready for the final test. Your score from the evaluation is the one you will be judged on. 

Note that the evaluation attempt will run three attempts in a row and your score will be the average result, so you should ensure your agent server keeps running through all three simulations.

The evaluation will have preset seeds.

## Quickstart

Clone the repository and enter the use case folder:

```cmd
git clone https://github.com/amboltio/Nordic-AI-Cup-2026
cd Nordic-AI-Cup-2026/survival-simulator
```

### Install
The simulator requires **Python 3.10 or newer**. We recommend installing the dependencies in a virtual environment so they do not interfere with your other projects.

Linux / macOS:
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Windows:
```cmd
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

Remember to activate the environment (`source .venv/bin/activate` or `.venv\Scripts\activate`) in every new terminal before running any of the scripts below.

To verify that the installation works, run a quick headless simulation:

```cmd
python -c "from local_playground import local_simulation; local_simulation(verbose=False)"
```

You should see a stream of `Score | Agents alive | Time` lines, ending with a `Game over!` message and the seed that was used.

# Testing locally
To test the simulation locally you can run [local_playground.py](local_playground.py). This can be used to get an idea of how the simulation works. It is recommended to use this file for any potential training with "verbose" set to False to run simulations faster.

# Run on server
You can serve your endpoint locally and test that everything starts without errors by running [agent_server.py](agent_server.py). Then open a browser and navigate to [http://localhost:9052](http://localhost:9052). You should see a message stating that the agent server is running. 
Feel free to change the `HOST` and `PORT` settings in [agent_server.py](agent_server.py).

To run a simulation on the server, you can run [simulation_server.py](simulation_server.py) while the endpoint is running.

The default movement logic for agents can be found in [dummy_agent_policy.py](src/utils/controllers/dummy_agent_policy.py).


### OBS
The simulation is deterministic as long as it runs on the same OS. If you want to test how a specific seed runs on the validation/evaluation server, you should test on a Linux machine.

To avoid bottlenecking the system, the server will wait for responses for up to 10 seconds. If no responses are received from the agent server within that time or if the accumulated wait time reaches 600 seconds, the run will end.
