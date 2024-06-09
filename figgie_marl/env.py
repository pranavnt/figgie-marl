import gymnasium as gym
import numpy as np
from typing import Dict, List, Tuple
import jax.numpy as jnp

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

        self.action_space = gym.spaces.Tuple([
            gym.spaces.Discrete(4),  # buy, bid, offer, pass
            gym.spaces.Discrete(4),  # suit index
            gym.spaces.Box(low=0, high=350, shape=(1,), dtype=np.int32)  # amount
        ])

        self.observation_space = gym.spaces.Dict({
            'player_id': gym.spaces.Discrete(num_players),
            'player_cards': gym.spaces.Box(low=0, high=12, shape=(4,), dtype=np.int32),
            'player_chips': gym.spaces.Box(low=0, high=1000, shape=(1,), dtype=np.int32),
            'opponent_card_counts': gym.spaces.Box(low=0, high=12, shape=(num_players-1, 4), dtype=np.int32),
            'bids': gym.spaces.Box(low=0, high=350, shape=(4,), dtype=np.int32),
            'offers': gym.spaces.Box(low=0, high=350, shape=(4,), dtype=np.int32),
            'completed_orders': gym.spaces.Box(low=0, high=350, shape=(20, 3), dtype=np.int32),
        })

    def _get_obs(self) -> Dict[str, np.ndarray]:
        obs = {
            'player_id': self.current_player,
        'player_cards': self.player_cards[self.current_player].reshape(-1),  # Change this line
        'player_chips': self.player_chips[self.current_player],
        'opponent_card_counts': np.delete(self.player_cards, self.current_player, axis=0),
        'bids': self.bids,
        'offers': self.offers,
        'completed_orders': self.completed_orders
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
        self.trading_time = 480  # 4 minutes in seconds

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

    def step(self, actions: Tuple[Tuple[int, int, int], ...]) -> Tuple[Dict[str, np.ndarray], List[int], bool, Dict]:
        for player_id, action in enumerate(actions):
            action_type, suit_index, amount = action
            if action_type == 0:  # buy
                if self.offers[suit_index] <= amount and amount <= self.player_chips[player_id]:
                    self.player_cards = self.player_cards.at[player_id, suit_index].add(1)
                    self.player_chips = self.player_chips.at[player_id].subtract(amount)
                    seller_id = np.where(self.offers == self.offers[suit_index])[0][0]
                    self._add_completed_order(player_id, seller_id, amount)
                    self.offers[suit_index] = 0
            elif action_type == 1:  # bid
                self.bids[suit_index] = max(self.bids[suit_index], amount)
            elif action_type == 2:  # offer
                if self.player_cards[player_id][suit_index] > 0:
                    self.offers[suit_index] = min(self.offers[suit_index], amount)

        self._process_completed_orders()

        self.trading_time -= 1
        done = (self.trading_time <= 0)

        if done:
            obs = self._get_obs()
            rewards = self._calculate_final_rewards()
            info = {}
        else:
            obs = self._get_obs()
            rewards = [0] * self.num_players
            info = {}

        self.current_player = (self.current_player + 1) % self.num_players

        return obs, rewards, done, info

    def _process_completed_orders(self):
        mask = self.bids > 0
        matched_suits = np.where(mask & (self.bids <= self.offers))[0]

        for suit_index in matched_suits:
            bid_player = np.where(self.bids == self.bids[suit_index])[0][0]
            offer_player = np.where(self.offers == self.offers[suit_index])[0][0]

            self.player_cards[bid_player][suit_index] += 1
            self.player_cards = self.player_cards.at[offer_player, suit_index].subtract(1)

            self.player_chips = self.player_chips.at[bid_player].subtract(self.bids[suit_index])
            self.player_chips = self.player_chips.at[offer_player].add(self.bids[suit_index])

            self._add_completed_order(bid_player, offer_player, self.bids[suit_index])

            self.bids[suit_index] = 0
            self.offers[suit_index] = 0

    def _add_completed_order(self, player_id_buy, player_id_sell, amount):
        self.completed_orders = np.roll(self.completed_orders, -1, axis=0)
        self.completed_orders[-1] = np.array([player_id_buy, player_id_sell, amount])

    def _calculate_final_rewards(self):
        rewards = [0] * self.num_players
        goal_suit_index = self.suits.index(self.goal_suit)

        for player_id in range(self.num_players):
            rewards[player_id] = self.player_cards[player_id][goal_suit_index] * 10

        max_goal_suit_cards = np.max(self.player_cards[:, goal_suit_index])
        winners = np.where(self.player_cards[:, goal_suit_index] == max_goal_suit_cards)[0]
        pot_bonus = (200 - self.goal_suit_cards * 10) // len(winners)
        for winner in winners:
            rewards[winner] += pot_bonus

        return rewards

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