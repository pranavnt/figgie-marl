# Figgie MARL Training Plan

## 1: Environment Design
- [x] see `figgie_marl/env.py`
- [ ] figure out what to do re not moving

# 2: Training loop
- [ ] Design and implement a neural network architecture for opponent modeling.
  - [ ] Input: Order book snapshots, player actions, and game state.
  - [ ] Output: Predicted distribution of cards in each opponent's hand.
- [ ] Evaluate this and update predictor with supervised learning
- [ ] Integrate the trained opponent modeling component into the agent's decision-making process.
- [ ] Train the agent with self-play (independent PPO probably)

## 3: Population-Based Training
- [ ] Train population of agents; maybe use prior checkpoints like ficticious coplay
- [ ] Clone policies of good agents if p(win) > threshold
- [ ] Not super sure about this; probably ask Natasha

## 4: Search
- [ ] Simulate and search through various action sequences based on the simulated opponents
- [ ] Integrate the search algorithm as final stage of policy

## 5: Evaluation and Iteration
- [ ] Play against humans, probably need more diversity due to adversarial human opponents
