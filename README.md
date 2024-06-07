# Figgie MARL

Training MARL bot to play figgie. Pretty similar to CICERO probably.

TL;DR:
- Break the game down into discrete timesteps; probably like 2/second, to roughly mimic human speed
- decentralized training probably?
- reasoning module:
  1. agents explictly model: other player's hands at every step, other player's values of each card, other player's actions, and the proabilities of goal suite
  2. simulate the game to search the best action for each player
  3. pick best action
