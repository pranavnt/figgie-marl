# Figgie MARL

## Figgie Rules
Figgie uses a deck of 40 cards: two suits consisting of 10 cards each, one suit consisting of 8 cards, and one of 12 cards. Which suit corresponds to which amount is random, and not known to the players beforehand.

The suit of 12 cards is the common suit. The goal suit is the suit of the same color as the common suit - i.e. if there are 12 Diamonds, then Hearts is the goal suit. (The goal suit may have either 8 cards or 10 cards).

Each player antes up $50 to play, creating a pot of $200. At the close of trading, each card of the goal suit is worth $10 from the pot, and then the remainder - either $100 or $120 - goes to the player with the most of the goal suit. (In a tie, the pot is split.)

At each step in the game, players issue bids/offers to buy a card. If a bid/offer is accepted, the all markets are cleared.
## MARL Environment
At each steps, agents get the following as observations:
- Their current set of cards (how many of each suite)
- The number of cards each opponent has
- The current order book (best offers)
- Past 20 completed orders

Agents then can do one of the following:
- Buy: Buy a card (if multiple buys on same offer, randomly)
- Bid: Place a bid for a given suite
- Offer: Place an offer for a given suite
- Pass: Not do anything
## Core Ideas
Our setup of the training process will be reasonably overengineered for Figgie. I still think it will be pretty hard to get superhuman performance on a reasonable compute budget without focusing specifically on Figgie. Here are my core ideas for setting this up:
- Predict the following initially (like how CICERO predicts the actions of the other players):
	- Card distribution of other players
	- Other player's actions
- Train with self-play using independent PPO
- Bootstrap agent with supervised learning on [hardcoded strategies](https://github.com/0xDub/figgie-auto)
- Inference time search:
	- Since each player is trained to simulate other players, you can roll out policies for a couple steps
	- Then, use trained value network to evaluate the best action at inference time
