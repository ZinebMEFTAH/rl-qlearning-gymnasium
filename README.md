# Reinforcement Learning — Q-Learning with Gymnasium

A from-scratch implementation of the **Q-Learning** algorithm, trained on several classic **Gymnasium** environments. The project covers the full loop: training agents, saving learned policies, and replaying them.

## Environments

- **Frozen Lake (4×4 & 8×8)** — pathfinding on a slippery grid while avoiding holes
- **Taxi-v3** — learning optimal pickup and drop-off logic
- **Cart Pole** — balancing a pole using continuous-state discretization

## Method

- **Q-Learning** — off-policy temporal-difference control
- **Exploration vs. exploitation** — ε-greedy strategy with decay
- **Continuous states** — discretization for Cart Pole

## Tech Stack

- Python, Gymnasium, NumPy, Pickle (policy persistence)

## Usage

```bash
# Train an agent
python trainAI.py

# Watch a trained agent play
python play.py
```

Trained-agent previews for each environment are included as images (`agent-*.png`).

## Project Structure

- `trainAI.py` — training loop
- `agent.py` — Q-Learning agent
- `play.py` — load and visualize a trained policy
- `agent-*.png` — sample results per environment

## Notes

Built to understand reinforcement learning fundamentals by implementing Q-Learning by hand rather than relying on a library.
