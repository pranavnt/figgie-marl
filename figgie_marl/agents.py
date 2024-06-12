import numpy as np
from typing import Tuple
from env import FiggieEnv

class BaseAgent:
    def __init__(self, player_id: int, num_players: int, num_suits: int):
        self.player_id = player_id
        self.num_players = num_players
        self.num_suits = num_suits

    def act(self, obs: np.ndarray) -> Tuple[int, int, int]:
        raise NotImplementedError

class TiltInventoryAgent(BaseAgent):
    def act(self, obs: np.ndarray) -> Tuple[int, int, int]:
        player_cards = obs[1:1+self.num_suits]
        player_chips = obs[1+self.num_suits]
        bids = obs[-7:-3]
        offers = obs[-3:]

        goal_suit_index = np.argmax(player_cards)
        non_goal_suit_indices = np.where(player_cards < player_cards[goal_suit_index])[0]

        if np.random.rand() < 0.6 and non_goal_suit_indices.size > 0:
            suit_index = np.random.choice(non_goal_suit_indices)
            if player_cards[suit_index] > 0 and offers[suit_index] >= 7:
                return 2, suit_index, offers[suit_index] - 1
        else:
            if offers[goal_suit_index] <= 5 and player_chips >= offers[goal_suit_index]:
                return 0, goal_suit_index, offers[goal_suit_index]
            elif bids[goal_suit_index] < 8 and player_chips >= bids[goal_suit_index] + 1:
                return 1, goal_suit_index, bids[goal_suit_index] + 1

        return 3, 0, 0

class SpreadAgent(BaseAgent):
    def act(self, obs: np.ndarray) -> Tuple[int, int, int]:
        player_cards = obs[1:1+self.num_suits]
        player_chips = obs[1+self.num_suits]
        bids = obs[-7:-3]
        offers = obs[-3:]
        completed_orders = obs[-63:-3].reshape(20, 3)

        average_inventory = np.mean(player_cards)

        for suit_index in range(self.num_suits):
            last_trade_price = completed_orders[:, 2][completed_orders[:, 1] == suit_index]
            last_trade_price = last_trade_price[-1] if last_trade_price.size > 0 else 0

            if player_cards[suit_index] > average_inventory:
                if last_trade_price > 0 and player_chips >= last_trade_price + 1:
                    return 2, suit_index, last_trade_price + 1
                elif offers[suit_index] > 7 and player_chips >= offers[suit_index] - 1:
                    return 2, suit_index, offers[suit_index] - 1
            else:
                if last_trade_price > 2 and player_chips >= last_trade_price - 1:
                    return 1, suit_index, last_trade_price - 1
                elif bids[suit_index] < 7 and player_chips >= bids[suit_index] + 2:
                    return 1, suit_index, bids[suit_index] + 2

        return 3, 0, 0

class SellerAgent(BaseAgent):
    def act(self, obs: np.ndarray) -> Tuple[int, int, int]:
        player_cards = obs[1:1+self.num_suits]
        bids = obs[-7:-3]
        time_left = obs[-1]

        for suit_index in range(self.num_suits):
            if player_cards[suit_index] > 0:
                if time_left >= 180:
                    if bids[suit_index] >= 6:
                        return 2, suit_index, bids[suit_index]
                    else:
                        return 2, suit_index, 8
                elif 120 < time_left < 180:
                    if bids[suit_index] >= 5:
                        return 2, suit_index, bids[suit_index]
                    else:
                        return 2, suit_index, 6
                elif 60 < time_left <= 120:
                    if bids[suit_index] >= 4:
                        return 2, suit_index, bids[suit_index]
                    else:
                        return 2, suit_index, 6
                else:
                    if bids[suit_index] >= 3:
                        return 2, suit_index, bids[suit_index]
                    else:
                        return 2, suit_index, 4

        return 3, 0, 0

class NoisyAgent(BaseAgent):
    def act(self, obs: np.ndarray) -> Tuple[int, int, int]:
        player_cards = obs[1:1+self.num_suits]
        player_chips = obs[1+self.num_suits]

        action_type = np.random.choice([0, 2])
        suit = np.random.randint(0, self.num_suits)

        if action_type == 0:
            price = np.random.randint(1, min(15, player_chips + 1))
            if player_cards[suit] < 4:
                return 0, suit, price
        else:
            price = np.random.randint(1, 15)
            if player_cards[suit] > 0:
                return 2, suit, price

        return 3, 0, 0

class PickOffAgent(BaseAgent):
    def act(self, obs: np.ndarray) -> Tuple[int, int, int]:
        player_cards = obs[1:1+self.num_suits]
        player_chips = obs[1+self.num_suits]
        bids = obs[-7:-3]
        offers = obs[-3:]
        time_left = obs[-1]

        open_price, close_price = self.get_prices(time_left)

        for suit_index in range(self.num_suits):
            if player_cards[suit_index] <= 2 and offers[suit_index] < open_price and player_chips >= offers[suit_index]:
                return 0, suit_index, offers[suit_index]

            if player_cards[suit_index] > 0:
                if bids[suit_index] >= close_price:
                    return 2, suit_index, bids[suit_index]
                elif offers[suit_index] > 5 and player_chips >= offers[suit_index] - 1:
                    return 2, suit_index, offers[suit_index] - 1

        return 3, 0, 0

    def get_prices(self, time_left: int) -> Tuple[int, int]:
        if time_left < 20:
            return 0, 0
        elif time_left < 40:
            return 2, 3
        elif time_left < 60:
            return 3, 4
        elif time_left < 120:
            return 4, 6
        else:
            return 5, 8

if __name__ == "__main__":
    num_players = 4
    num_rounds = 480
    num_games = 10

    # Define the agents
    agents = [
        PickOffAgent(0, num_players, 4),
        TiltInventoryAgent(1, num_players, 4),
        NoisyAgent(2, num_players, 4),
        SellerAgent(3, num_players, 4)
    ]

    # Run multiple games
    for game_id in range(num_games):
        env = FiggieEnv(num_players)
        obs = env.reset()

        round_num = 0
        done = False
        while not done:
            actions = []
            for i in range(num_players):
                agent = agents[i]
                action = agent.act(obs[i])
                actions.append(action)

            obs, rewards, done, info = env.step(tuple(actions))

            round_num += 1
            if round_num >= num_rounds:
                done = True

        print(f"Game {game_id+1} - Final balances: {rewards}")
