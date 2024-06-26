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
        self.ante = 50
        self.pot = self.num_players * self.ante

        self.action_space = gym.spaces.Tuple([
            gym.spaces.Discrete(4),  # buy, bid, offer, pass
            gym.spaces.Discrete(4),  # suit index
            gym.spaces.Box(low=0, high=350, shape=(1,), dtype=np.int32)  # amount
        ])

        self.obs_size = 1 + self.num_suits + 1 + (self.num_players - 1) * self.num_suits + self.num_suits * 2 + 20 * 3
        self.observation_space = gym.spaces.Box(low=0, high=350, shape=(self.num_players, self.obs_size), dtype=np.int32)

    def _get_obs(self) -> jnp.ndarray:
        obs = np.zeros((self.num_players, self.obs_size), dtype=np.int32)

        for player_id in range(self.num_players):
            offset = 0
            obs[player_id, offset] = player_id
            offset += 1

            obs[player_id, offset:offset+self.num_suits] = self.player_cards[player_id]
            offset += self.num_suits

            obs[player_id, offset] = self.player_chips[player_id]
            offset += 1

            opponent_card_counts = np.delete(self.player_cards, player_id, axis=0).flatten()
            obs[player_id, offset:offset+len(opponent_card_counts)] = opponent_card_counts
            offset += len(opponent_card_counts)

            obs[player_id, offset:offset+self.num_suits] = self.bids
            offset += self.num_suits

            obs[player_id, offset:offset+self.num_suits] = self.offers
            offset += self.num_suits

            obs[player_id, offset:] = self.completed_orders.flatten()

        return jnp.array(obs)

    def reset(self) -> jnp.ndarray:
        self.player_cards = np.zeros((self.num_players, self.num_suits), dtype=np.int32)
        if not hasattr(self, 'player_chips'):
            self.player_chips = np.full((self.num_players,), 300, dtype=np.int32)
        else:
            self.player_chips -= self.ante
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

    def step(self, actions: Tuple[Tuple[int, int, int], ...]) -> Tuple[jnp.ndarray, List[int], bool, Dict]:
        for player_id, action in enumerate(actions):
            action_type, suit_index, amount = action
            if action_type == 0:  # buy
                if self.offers[suit_index] > 0 and amount >= self.offers[suit_index] and self.player_chips[player_id] >= amount:
                    seller_id = np.where(self.offers == self.offers[suit_index])[0][0]
                    if self.player_cards[seller_id][suit_index] > 0:
                        self.player_cards[player_id][suit_index] += 1
                        self.player_chips[player_id] -= amount
                        self.player_cards[seller_id][suit_index] -= 1
                        self.player_chips[seller_id] += amount
                        self._add_completed_order(player_id, seller_id, amount)
                        self.offers[suit_index] = 0
            elif action_type == 1:  # bid
                if amount > self.bids[suit_index] and self.player_chips[player_id] >= amount:
                    self.bids[suit_index] = amount
            elif action_type == 2:  # offer
                if self.player_cards[player_id][suit_index] > 0 and (self.offers[suit_index] == 0 or amount < self.offers[suit_index]):
                    self.offers[suit_index] = amount

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

            trade_amount = min(self.bids[suit_index], self.offers[suit_index])

            if self.player_chips[bid_player] >= trade_amount and self.player_cards[offer_player][suit_index] > 0:
                self.player_cards[bid_player][suit_index] += 1
                self.player_cards[offer_player][suit_index] -= 1

                self.player_chips[bid_player] -= trade_amount
                self.player_chips[offer_player] += trade_amount

                self._add_completed_order(bid_player, offer_player, trade_amount)

                self.bids[suit_index] = 0
                self.offers[suit_index] = 0

    def _add_completed_order(self, player_id_buy, player_id_sell, amount):
        self.completed_orders = np.roll(self.completed_orders, -1, axis=0)
        self.completed_orders[-1] = np.array([player_id_buy, player_id_sell, amount])

    def _calculate_final_rewards(self):
        rewards = [0] * self.num_players
        goal_suit_index = self.suits.index(self.goal_suit)

        total_goal_suit_cards = np.sum(self.player_cards[:, goal_suit_index])

        for player_id in range(self.num_players):
            num_goal_suit_cards = self.player_cards[player_id][goal_suit_index]
            self.player_chips[player_id] += 200 * num_goal_suit_cards / total_goal_suit_cards

        return self.player_chips.tolist()

    def render(self, mode='human'):
        if mode == 'human':
            pass
        else:
            super(FiggieEnv, self).render(mode=mode)

    def close(self):
        pass