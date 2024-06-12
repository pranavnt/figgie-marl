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
rng = jax.random.PRNGKey(0)
init_obs = env.observation_space.sample()

params = agent.init(rng, init_obs[0])

tx = optax.chain(optax.clip_by_global_norm(config["max_grad_norm"]), optax.adam(learning_rate=config["learning_rate"]))
train_state = TrainState.create(apply_fn=agent.apply, params=params, tx=tx)

def update_step(rng, train_state, observations, actions, rewards, dones, values):
    def loss_fn(params):
        (pi_action_type, pi_suit, pi_amount), value = agent.apply(params, observations)

        log_prob_action_type = pi_action_type.log_prob(actions[:, 0])
        log_prob_suit = pi_suit.log_prob(actions[:, 1])
        log_prob_amount = pi_amount.log_prob(actions[:, 2])
        log_prob = log_prob_action_type + log_prob_suit + log_prob_amount

        value_loss = jnp.mean(jnp.square(value - rewards))

        advantage = rewards - values
        policy_loss = -jnp.mean(log_prob * advantage)

        entropy_bonus = config["entropy_coef"] * (pi_action_type.entropy() + pi_suit.entropy() + pi_amount.entropy()).mean()

        total_loss = policy_loss + config["vf_coef"] * value_loss - entropy_bonus

        return total_loss, (policy_loss, value_loss, entropy_bonus)

    grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
    (loss, (policy_loss, value_loss, entropy_bonus)), grads = grad_fn(train_state.params)
    train_state = train_state.apply_gradients(grads=grads)

    return train_state, loss, policy_loss, value_loss, entropy_bonus

for epoch in range(config["num_epochs"]):
    obs = env.reset()
    done = False

    replay_buffer = []
    # obses
    # actions


    while not done:
        rng, _rng = jax.random.split(rng)

        actions = []
        values = []

        for i in range(config["num_players"]):
            player_obs = obs[i]
            action, value = agent.act(train_state.params, player_obs, _rng)
            actions.append(action)
            values.append(value)

        actions = jnp.stack(actions, axis=0)
        values = jnp.stack(values, axis=0)

        next_obs, rewards, done, _ = env.step(actions)

        train_state, loss, policy_loss, value_loss, entropy_bonus = update_step(
            rng, train_state, obs, actions, rewards, done, value.squeeze()
        )

        obs = next_obs

    print(f"Epoch: {epoch+1}, Loss: {loss:.3f}, Policy Loss: {policy_loss:.3f}, Value Loss: {value_loss:.3f}, Entropy Bonus: {entropy_bonus:.3f}")
