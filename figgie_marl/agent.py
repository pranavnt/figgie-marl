import jax
import jax.numpy as jnp
import flax.linen as nn
from env import FiggieEnv

class LSTMNetwork(nn.Module):
    hidden_dim: int

    @nn.compact
    def __call__(self, x):
        lstm = nn.LSTMCell(self.hidden_dim)
        carry = lstm.initialize_carry(jax.random.PRNGKey(0), (1,) + x.shape[1:])
        (new_c, new_h), _ = lstm(carry, x)
        return new_h

class Agent(nn.Module):
    num_players: int
    num_suits: int
    hidden_dim: int

    @nn.compact
    def __call__(self, obs, predict_cards=False):
        opponent_card_dist = nn.Dense(self.hidden_dim)(obs)
        opponent_card_dist = nn.relu(opponent_card_dist)
        opponent_card_dist = nn.Dense(self.hidden_dim * 2)(opponent_card_dist)
        opponent_card_dist = nn.relu(opponent_card_dist)
        opponent_card_dist = nn.Dense((self.num_players - 1) * self.num_suits)(opponent_card_dist)
        opponent_card_dist = nn.softmax(opponent_card_dist, axis=-1)
        opponent_card_dist = jnp.reshape(opponent_card_dist, (-1, self.num_players - 1, self.num_suits))

        if predict_cards:
            return opponent_card_dist

        features = jnp.concatenate([obs, jnp.ravel(opponent_card_dist)])

        actor = LSTMNetwork(self.hidden_dim)(jnp.expand_dims(features, axis=0))
        actor = jnp.squeeze(actor, axis=0)
        actor = nn.Dense(self.hidden_dim * 2)(actor)
        actor = nn.relu(actor)
        action_type_logits = nn.Dense(4)(actor)

        suit_logits = nn.Dense(self.num_suits)(actor)

        amount_mu = nn.Dense(1)(actor)
        amount_sigma = nn.softplus(nn.Dense(1)(actor)) + 1e-6

        critic = nn.Dense(self.hidden_dim)(features)
        critic = nn.relu(critic)
        critic = nn.Dense(self.hidden_dim * 2)(critic)
        critic = nn.relu(critic)
        critic = nn.Dense(self.hidden_dim * 2)(critic)
        critic = nn.relu(critic)
        value = nn.Dense(1)(critic)

        return action_type_logits, suit_logits, amount_mu, amount_sigma, value

    def act(self, params, obs, rng_key):
        action_type_logits, suit_logits, amount_mu, amount_sigma, value = self.apply(params, obs)

        action_type_key, suit_key, amount_key = jax.random.split(rng_key, 3)
        action_type = jax.random.categorical(action_type_key, action_type_logits)
        suit = jax.random.categorical(suit_key, suit_logits)
        amount = jax.random.normal(amount_key) * amount_sigma + amount_mu
        player_balance = obs[1 + self.num_suits]
        amount = jnp.clip(amount, 0, player_balance).astype(jnp.int32)

        return jnp.array([action_type, suit, amount[0]]), value