import gymnasium as gym
import numpy as np
from typing import Dict, List, Tuple

class FiggieEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self, num_players: int = 4):
        super(FiggieEnv, self).__init__()

        self.num_players = num_players
        self.num_cards = 40
        self.num_suits = 4
        self.suits = ['♠', '♣', '♥', '♦']
        self.common_suit = None
        self.goal_suit = None
        self.goal_suit_cards = None

        self.action_space = gym.spaces.Discrete(4)  # buy, bid, offer, pass

        self.observation_space = gym.spaces.Dict({
            'player_id': gym.spaces.Discrete(num_players),
            'player_cards': gym.spaces.Box(low=0, high=12, shape=(4,), dtype=np.int32),
            'player_chips': gym.spaces.Box(low=0, high=1000, shape=(1,), dtype=np.int32),
            'opponent_card_counts': gym.spaces.Box(low=0, high=12, shape=(num_players-1, 4), dtype=np.int32),
            'bids': gym.spaces.Box(low=0, high=350, shape=(4,), dtype=np.int32),
            'offers': gym.spaces.Box(low=0, high=350, shape=(4,), dtype=np.int32),
            'completed_orders': gym.spaces.Box(low=0, high=350, shape=(20, 3), dtype=np.int32),  # suit, price, quantity
        })

    def _get_obs(self) -> Dict[str, np.ndarray]:
        obs = {
            'player_id': self.current_player,
            'player_cards': self.player_cards[self.current_player],
            'player_chips': self.player_chips[self.current_player],
            'opponent_card_counts': np.delete(self.player_cards, self.current_player, axis=0),
            'bids': self.bids,
            'offers': self.offers,
            'completed_orders': self.completed_orders,
        }
        return obs

    def reset(self) -> Dict[str, np.ndarray]:
        self.player_cards = np.zeros((self.num_players, self.num_suits), dtype=np.int32)
        self.player_chips = np.full((self.num_players,), 300, dtype=np.int32)  # $350 - $50 ante
        self.bids = np.zeros(self.num_suits, dtype=np.int32)
        self.offers = np.zeros(self.num_suits, dtype=np.int32)
        self.completed_orders = np.zeros((20, 3), dtype=np.int32)

        self.deck = self._create_deck()
        self.common_suit = [suit for suit, count in self.deck if count == 12][0]
        self.goal_suit = '♥' if self.common_suit == '♦' else '♦' if self.common_suit == '♥' else '♠' if self.common_suit == '♣' else '♣'
        self.goal_suit_cards = [count for suit, count in self.deck if suit == self.goal_suit][0]

        self._deal_cards()

        self.current_player = 0
        self.trading_time = 240  # 4 minutes in seconds

        return self._get_obs()

    def _create_deck(self) -> List[Tuple[str, int]]:
        deck = []
        suit_counts = np.random.choice([8, 10, 10, 12], size=4, replace=False)
        for suit, count in zip(self.suits, suit_counts):
            deck.append((suit, count))
        return deck

    def _deal_cards(self):
        cards = [(suit, 1) for suit, count in self.deck for _ in range(count)]
        np.random.shuffle(cards)
        for i, (suit, _) in enumerate(cards):
            player = i % self.num_players
            suit_index = self.suits.index(suit)
            self.player_cards[player][suit_index] += 1

    def step(self, action: int) -> Tuple[Dict[str, np.ndarray], int, bool, Dict]:
        if action == 0:  # buy
            self._process_buy_action()
        elif action == 1:  # bid
            self._process_bid_action()
        elif action == 2:  # offer
            self._process_offer_action()
        # else: pass

        self._process_completed_orders()

        self.trading_time -= 1
        done = (self.trading_time <= 0)

        if not done:
            self.current_player = (self.current_player + 1) % self.num_players

        obs = self._get_obs()
        reward = self._calculate_reward() if done else 0
        info = {}

        return obs, reward, done, info

    def _process_buy_action(self):
        # Implement buy action logic
        pass

    def _process_bid_action(self):
        # Implement bid action logic
        pass

    def _process_offer_action(self):
        # Implement offer action logic
        pass

    def _process_completed_orders(self):
        # Implement completed order processing logic
        pass

    def _calculate_reward(self) -> int:
        rewards = np.zeros(self.num_players, dtype=np.int32)
        goal_suit_index = self.suits.index(self.goal_suit)

        for player in range(self.num_players):
            rewards[player] = self.player_cards[player][goal_suit_index] * 10

        max_goal_suit_cards = np.max(self.player_cards[:, goal_suit_index])
        winners = np.where(self.player_cards[:, goal_suit_index] == max_goal_suit_cards)[0]
        pot_bonus = (200 - self.goal_suit_cards * 10) // len(winners)
        rewards[winners] += pot_bonus

        return rewards[self.current_player]

    def render(self, mode='human'):
        if mode == 'human':
            print(f"Player: {self.current_player}")
            print(f"Player Chips: {self.player_chips}")
            print(f"Player Cards: {self.player_cards}")
            print(f"Bids: {self.bids}")
            print(f"Offers: {self.offers}")
            print(f"Completed Orders: {self.completed_orders}")
        else:
            super(FiggieEnv, self).render(mode=mode)

    def close(self):
        pass
