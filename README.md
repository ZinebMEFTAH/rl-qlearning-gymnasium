# Reinforcement Learning: Q-Learning with Gymnasium

Implementation of the Q-Learning algorithm from scratch to train autonomous agents in various simulated environments.

## Environments
- **Frozen Lake (4x4 & 8x8)**: Pathfinding on a slippery grid while avoiding holes.
- **Taxi-v3**: Optimizing pickup and drop-off logic for a virtual taxi.
- **Cart Pole**: Balancing a pole on a cart using state discretization.

## Algorithm
- **Q-Learning**: Off-policy temporal difference control.
- **Exploration vs. Exploitation**: Implemented via an epsilon-greedy strategy.
- **State Handling**: Continuous state space discretization for the Cart Pole environment.

## Tech Stack
- Python, Gymnasium, Numpy, Pickle.

## Usage
To train the agent:
`python train.py --env FrozenLake-v1`
To watch the trained agent:
`python play.py --env FrozenLake-v1 -i`
