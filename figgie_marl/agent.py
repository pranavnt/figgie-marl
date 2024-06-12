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
    def __call__(self, obs):
        opponent_card_dist = nn.Dense(self.hidden_dim)(obs)
        opponent_card_dist = nn.relu(opponent_card_dist)
        opponent_card_dist = nn.Dense(self.hidden_dim * 2)(opponent_card_dist)
        opponent_card_dist = nn.relu(opponent_card_dist)
        opponent_card_dist = nn.Dense(self.num_suits * (self.num_players - 1))(opponent_card_dist)
        opponent_card_dist = nn.softmax(opponent_card_dist, axis=-1)
        opponent_card_dist = jnp.reshape(opponent_card_dist, (self.num_players - 1, self.num_suits))

        opponent_actions = nn.Dense(self.hidden_dim)(obs)
        opponent_actions = nn.relu(opponent_actions)
        opponent_actions = nn.Dense(self.hidden_dim * 2)(opponent_actions)
        opponent_actions = nn.relu(opponent_actions)
        opponent_actions = nn.Dense(4 * (self.num_players - 1))(opponent_actions)
        opponent_actions = nn.softmax(opponent_actions, axis=-1)
        opponent_actions = jnp.reshape(opponent_actions, (self.num_players - 1, 4))

        features = jnp.concatenate([obs, jnp.ravel(opponent_card_dist), jnp.ravel(opponent_actions)])

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
        action_type_logits, suit_logits, amount_mu, amount_sigma, _ = self.apply(params, obs)

        action_type_key, suit_key, amount_key = jax.random.split(rng_key, 3)
        action_type = jax.random.categorical(action_type_key, action_type_logits)
        suit = jax.random.categorical(suit_key, suit_logits)
        amount = jax.random.normal(amount_key) * amount_sigma + amount_mu
        player_balance = obs[1 + self.num_suits]
        amount = jnp.clip(amount, 0, player_balance).astype(jnp.int32)

        return jnp.array([action_type, suit, amount[0]])


if __name__ == "__main__":
    num_players = 4
    num_suits = 4
    hidden_dim = 128

    env = FiggieEnv(num_players=num_players)

    rng_key = jax.random.PRNGKey(0)
    agents = []
    agent_params = []
    for _ in range(num_players):
        agent = Agent(num_players=num_players, num_suits=num_suits, hidden_dim=hidden_dim)
        rng_key, init_key = jax.random.split(rng_key)
        params = agent.init(init_key, env.observation_space.sample()[0])
        agents.append(agent)
        agent_params.append(params)

    obs = env.reset()

    done = False
    round_num = 0
    while not done:
        actions = []
        for i in range(num_players):
            player_obs = obs[i]

            rng_key, subkey = jax.random.split(rng_key)
            action = agents[i].act(agent_params[i], player_obs, subkey)
            actions.append(action)

        actions = tuple(actions)
        obs, rewards, done, info = env.step(actions)

        if round_num % 60 == 0:
            print(round_num)
        round_num += 1

    print("Game over!")
    print(f"Final rewards: {rewards}")