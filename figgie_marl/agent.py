import jax
import jax.numpy as jnp
import flax.linen as nn
from env import FiggieEnv

class Agent(nn.Module):
    num_players: int
    num_suits: int
    hidden_dim: int

    @nn.compact
    def __call__(self, obs):
        player_cards = obs['player_cards']
        opponent_card_counts = obs['opponent_card_counts']
        bids = obs['bids']
        offers = obs['offers']
        completed_orders = obs['completed_orders']

        opponent_card_counts_flat = jnp.ravel(opponent_card_counts)
        opponent_card_dist = nn.Dense(self.num_suits * (self.num_players - 1))(opponent_card_counts_flat)
        opponent_card_dist = jnp.reshape(opponent_card_dist, (self.num_players - 1, self.num_suits))
        opponent_card_dist = nn.softmax(opponent_card_dist, axis=-1)

        opponent_actions = nn.Dense(4 * (self.num_players - 1))(jnp.ravel(opponent_card_counts))
        opponent_actions = jnp.reshape(opponent_actions, (self.num_players - 1, 4))
        opponent_actions = nn.softmax(opponent_actions, axis=-1)

        goal_suit_pred = nn.Dense(self.num_suits)(jnp.concatenate([player_cards, jnp.ravel(opponent_card_counts)]))
        goal_suit_pred = nn.softmax(goal_suit_pred)

        features = jnp.concatenate([
          player_cards,
          jnp.ravel(opponent_card_dist),
          jnp.ravel(opponent_actions),
          goal_suit_pred,
          bids,
            offers,
            jnp.ravel(completed_orders)
        ])

        actor = nn.Dense(self.hidden_dim, kernel_init=nn.initializers.zeros)(features)
        actor = nn.relu(actor)
        actor = nn.Dense(self.hidden_dim)(actor)
        actor = nn.relu(actor)
        action_logits = nn.Dense(4)(actor)
        action_probs = nn.softmax(action_logits)

        critic = nn.Dense(self.hidden_dim)(features)
        critic = nn.relu(critic)
        critic = nn.Dense(self.hidden_dim)(critic)
        critic = nn.relu(critic)
        value = nn.Dense(1)(critic)

        return action_probs, value, opponent_card_dist, opponent_actions, goal_suit_pred

    def act(self, params, obs, rng_key):
        action_probs, _, _, _, _ = self.apply(params, obs)
        action = jax.random.categorical(rng_key, jnp.log(action_probs))
        return action

if __name__ == "__main__":
    num_players = 4
    num_suits = 4
    hidden_dim = 64

    env = FiggieEnv(num_players=num_players)

    rng_key = jax.random.PRNGKey(0)
    agents = []
    agent_params = []
    for _ in range(num_players):
        agent = Agent(num_players=num_players, num_suits=num_suits, hidden_dim=hidden_dim)
        rng_key, init_key = jax.random.split(rng_key)
        params = agent.init(init_key, {'player_cards': jnp.zeros((num_suits,)), 'opponent_card_counts': jnp.zeros((num_players-1, num_suits)), 'bids': jnp.zeros((num_suits,)), 'offers': jnp.zeros((num_suits,)), 'completed_orders': jnp.zeros((num_suits,))})
        agents.append(agent)
        agent_params.append(params)

   # Reset the environment
    obs = env.reset()

    done = False
    while not done:
        actions = []
        for i in range(num_players):
            player_obs = {}
            for k, v in obs.items():
                if isinstance(v, jnp.ndarray) and v.ndim > 1:
                    player_obs[k] = v[i]
                elif isinstance(v, jnp.ndarray) and v.ndim == 1 and len(v) == num_players:
                    player_obs[k] = v[i]
                else:
                    player_obs[k] = v

            rng_key, subkey = jax.random.split(rng_key)
            action = agents[i].act(agent_params[i], player_obs, subkey)
            actions.append(action)

        actions = tuple(actions)
        obs, rewards, done, info = env.step(actions)

        env.render()
        print(f"Rewards: {rewards}")

    print("Game over!")
    print(f"Final rewards: {rewards}")