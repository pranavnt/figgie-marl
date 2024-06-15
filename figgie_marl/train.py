import jax
import jax.numpy as jnp
import numpy as np
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
    "card_pred_learning_rate": 1e-3,
    "max_grad_norm": 0.5,
    "vf_coef": 0.5,
    "entropy_coef": 0.01,
    "num_epochs": 1000,
}

env = FiggieEnv(num_players=config["num_players"])


agent = Agent(num_players=config["num_players"], num_suits=config["num_suits"], hidden_dim=config["hidden_dim"])

rng_key = jax.random.PRNGKey(np.random.randint(0, 2**32))
rng_key, init_key = jax.random.split(rng_key)
params = agent.init(init_key, env.observation_space.sample()[0])

tx = optax.chain(optax.clip_by_global_norm(config["max_grad_norm"]), optax.adam(learning_rate=config["learning_rate"]))
tx_card_pred = optax.adam(learning_rate=config["card_pred_learning_rate"])  # Separate optimizer for card prediction
train_state = TrainState.create(apply_fn=agent.apply, params=params, tx=tx)
train_state_card_pred = TrainState.create(apply_fn=agent.apply, params=params, tx=tx_card_pred)

def loss_fn(params):
    observations = jnp.stack(replay_buffer["observations"], axis=0)
    actions = jnp.stack(replay_buffer["actions"], axis=0)
    rewards = jnp.array(replay_buffer["rewards"])
    dones = jnp.stack(replay_buffer["dones"], axis=0)
    values = jnp.stack(replay_buffer["values"], axis=0)
    log_probs = jnp.stack(replay_buffer["log_probs"], axis=0)
    opponent_cards = jnp.stack(replay_buffer["opponent_cards"], axis=0)

    value_loss = jnp.mean(jnp.square(values - rewards))
    advantage = rewards - values
    policy_loss = -jnp.mean(log_probs * advantage)
    entropy_bonus = config["entropy_coef"] * -jnp.mean(log_probs)

    # Card prediction loss
    card_pred_loss = 0
    for i in range(config["num_players"]):
        pred_card_dist = agent.apply(params, observations[:, i], predict_cards=True)
        true_card_dist = opponent_cards[:, i].reshape((-1, config["num_players"]-1, config["num_suits"]))
        card_pred_loss += jnp.mean(optax.softmax_cross_entropy(pred_card_dist, true_card_dist))

    total_loss = policy_loss + config["vf_coef"] * value_loss - entropy_bonus + card_pred_loss
    return total_loss, (policy_loss, value_loss, entropy_bonus, card_pred_loss)

def train_step(train_state_policy, train_state_card_pred, params):
    (loss, (policy_loss, value_loss, entropy_bonus, card_pred_loss)), grads = jax.value_and_grad(loss_fn, has_aux=True)(params)

    policy_grads, card_pred_grads = jax.tree_map(lambda x: x[0], grads), jax.tree_map(lambda x: x[1], grads)

    train_state_policy = train_state_policy.apply_gradients(grads=policy_grads)
    train_state_card_pred = train_state_card_pred.apply_gradients(grads=card_pred_grads)

    return train_state_policy, train_state_card_pred, loss, (policy_loss, value_loss, entropy_bonus, card_pred_loss)

train_step = jax.jit(train_step)

for epoch in range(config["num_epochs"]):
    obs = env.reset()
    done = False

    replay_buffer = {
        "observations": [],
        "actions": [],
        "rewards": [],
        "dones": [],
        "values": [],
        "log_probs": [],
        "opponent_cards": [],
    }

    while not done:
        rng_key, _ = jax.random.split(rng_key)

        actions = []
        values = []
        log_probs = []

        for i in range(config["num_players"]):
            player_obs = obs[i]
            (action_type, suit, amount), value = agent.act(train_state.params, player_obs, _rng)
            action = jnp.array([action_type, suit, amount])
            actions.append(action)
            values.append(value)

            action_type_logits, suit_logits, amount_mu, amount_sigma, _ = agent.apply(train_state.params, player_obs)
            action_type_dist = distrax.Categorical(logits=action_type_logits)
            suit_dist = distrax.Categorical(logits=suit_logits)
            amount_dist = distrax.Normal(loc=amount_mu, scale=amount_sigma)

            log_prob_action_type = action_type_dist.log_prob(action_type)
            log_prob_suit = suit_dist.log_prob(suit)
            log_prob_amount = amount_dist.log_prob(amount)
            log_prob = log_prob_action_type + log_prob_suit + log_prob_amount
            log_probs.append(log_prob)

        actions = jnp.stack(actions, axis=0)
        values = jnp.stack(values, axis=0)
        log_probs = jnp.stack(log_probs, axis=0)

        next_obs, rewards, done, _ = env.step(tuple(actions))

        replay_buffer["observations"].append(obs)
        replay_buffer["actions"].append(actions)
        replay_buffer["rewards"].append(rewards)
        replay_buffer["dones"].append(done)
        replay_buffer["values"].append(values.squeeze())
        replay_buffer["log_probs"].append(log_probs.squeeze())

        # Store true opponent card counts
        opponent_cards = []
        for i in range(config["num_players"]):
            opp_cards = jnp.delete(obs[:, 1:1+config["num_suits"]], i, axis=0).flatten()
            opponent_cards.append(opp_cards)
        replay_buffer["opponent_cards"].append(jnp.stack(opponent_cards, axis=0))

        obs = next_obs

    train_state, train_state_card_pred, loss, (policy_loss, value_loss, entropy_bonus, card_pred_loss) = train_step(train_state, train_state_card_pred, train_state.params)

    print(f"Epoch {epoch+1}, Loss: {loss:.3f}, Policy Loss: {policy_loss:.3f}, Value Loss: {value_loss:.3f}, Entropy Bonus: {entropy_bonus:.3f}, Card Pred Loss: {card_pred_loss:.3f}")