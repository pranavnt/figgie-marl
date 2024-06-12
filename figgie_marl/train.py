import jax
import jax.numpy as jnp
import flax.linen as nn
import optax
from flax.training.train_state import TrainState
import distrax
from typing import Sequence, Dict
import functools
from env import FiggieEnv
from agent import Agent

config = {
    "num_players": 4,
    "num_suits": 4,
    "hidden_dim": 128,
    "learning_rate": 3e-4,
    "max_grad_norm": 0.5,
    "vf_coef": 0.5,
    "entropy_coef": 0.01,
    "num_epochs": 1000,
}

env = FiggieEnv(num_players=config["num_players"])

rng_key = jax.random.PRNGKey(0)

agent = Agent(num_players=config["num_players"], num_suits=config["num_suits"], hidden_dim=config["hidden_dim"])
rng_key, init_key = jax.random.split(rng_key)
params = agent.init(init_key, env.observation_space.sample()[0])
tx = optax.chain(optax.clip_by_global_norm(config["max_grad_norm"]), optax.adam(learning_rate=config["learning_rate"]))
train_state = TrainState.create(apply_fn=agent.apply, params=params, tx=tx)

for epoch in range(config["num_epochs"]):
    obs = env.reset()
    done = False

    replay_buffer = {
        "observations": [],
        "actions": [],
        "rewards": [],
        "dones": [],
        "values": [],
        "actual_cards": [],
    }

    while not done:
        rng_key, _rng = jax.random.split(rng_key)

        actions = []
        values = []
        actual_cards = []

        for i in range(config["num_players"]):
            player_obs = obs[i]
            action, value = agent.act(train_state.params, player_obs, _rng)
            actions.append(action)
            values.append(value)
            actual_cards.append(player_obs[1:1+config["num_suits"]])

        actions = jnp.stack(actions, axis=0)
        values = jnp.stack(values, axis=0)
        actual_cards = jnp.stack(actual_cards, axis=0)

        next_obs, rewards, done, _ = env.step(actions)


        replay_buffer["observations"].append(obs)
        replay_buffer["actions"].append(actions)
        replay_buffer["rewards"].append(rewards)
        replay_buffer["dones"].append(done)
        replay_buffer["values"].append(values.squeeze())
        replay_buffer["actual_cards"].append(actual_cards)

        obs = next_obs

    # Perform update
    def loss_fn(params):
        observations = jnp.stack(replay_buffer["observations"], axis=0)
        actions = jnp.stack(replay_buffer["actions"], axis=0)
        rewards = jnp.array(replay_buffer["rewards"])
        dones = jnp.stack(replay_buffer["dones"], axis=0)
        values = jnp.stack(replay_buffer["values"], axis=0)
        actual_cards = jnp.stack(replay_buffer["actual_cards"], axis=0)

        (pi_action_type, pi_suit, pi_amount), value, predicted_cards = agent.apply(params, observations)

        log_prob_action_type = pi_action_type.log_prob(actions[:, :, 0])
        log_prob_suit = pi_suit.log_prob(actions[:, :, 1])
        log_prob_amount = pi_amount.log_prob(actions[:, :, 2])
        log_prob = log_prob_action_type + log_prob_suit + log_prob_amount

        value_loss = jnp.mean(jnp.square(value - rewards))

        advantage = rewards - values
        policy_loss = -jnp.mean(log_prob * advantage)

        entropy_bonus = config["entropy_coef"] * (pi_action_type.entropy() + pi_suit.entropy() + pi_amount.entropy()).mean()

        card_prediction_loss = -jnp.sum(actual_cards * jnp.log(predicted_cards), axis=-1).mean()

        total_loss = policy_loss + config["vf_coef"] * value_loss - entropy_bonus + card_prediction_loss

        return total_loss, (policy_loss, value_loss, entropy_bonus, card_prediction_loss)

    grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
    (loss, (policy_loss, value_loss, entropy_bonus, card_prediction_loss)), grads = grad_fn(train_state.params)
    train_state = train_state.apply_gradients(grads=grads)

    print(f"Epoch {epoch+1}, Loss: {loss:.3f}, Policy Loss: {policy_loss:.3f}, Value Loss: {value_loss:.3f}, Entropy Bonus: {entropy_bonus:.3f}, Card Prediction Loss: {card_prediction_loss:.3f}")