# figgie_marl/agent.py
import jax
import jax.numpy as jnp
import flax.linen as nn
import optax
from typing import Tuple
from figgie_marl.utils import ReplayBuffer, PPOParams

class PolicyNetwork(nn.Module):
    action_dim: int

    @nn.compact
    def __call__(self, x):
        x = nn.Dense(64)(x)
        x = nn.relu(x)
        x = nn.Dense(64)(x)
        x = nn.relu(x)
        x = nn.Dense(self.action_dim)(x)
        return nn.log_softmax(x)

class ValueNetwork(nn.Module):
    @nn.compact
    def __call__(self, x):
        x = nn.Dense(64)(x)
        x = nn.relu(x)
        x = nn.Dense(64)(x)
        x = nn.relu(x)
        x = nn.Dense(1)(x)
        return x

class SuitDistributionPredictor(nn.Module):
    num_players: int
    num_suits: int

    @nn.compact
    def __call__(self, x):
        x = nn.Dense(64)(x)
        x = nn.relu(x)
        x = nn.Dense(64)(x)
        x = nn.relu(x)
        x = nn.Dense(self.num_players * self.num_suits)(x)
        return nn.softmax(x.reshape((self.num_players, self.num_suits)), axis=-1)

class Agent:
    def __init__(self, id: int, obs_dim: int, action_dim: int, num_players: int, num_suits: int, lr: float, gamma: float, lambda_: float) -> None:
        self.id = id
        self.policy_network = PolicyNetwork(action_dim)
        self.value_network = ValueNetwork()
        self.suit_distribution_predictor = SuitDistributionPredictor(num_players, num_suits)
        self.policy_optimizer = optax.adam(lr)
        self.value_optimizer = optax.adam(lr)
        self.suit_distribution_optimizer = optax.adam(lr)
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.lambda_ = lambda_

    def act(self, obs: jnp.array, rng_key: jnp.array) -> Tuple[jnp.array, jnp.array]:
        logits = self.policy_network.apply(self.policy_params, obs)
        action = jax.random.categorical(rng_key, logits)
        logp = jnp.take_along_axis(logits, jnp.expand_dims(action, axis=-1), axis=-1)
        return action, logp

    def evaluate(self, obs: jnp.array) -> Tuple[jnp.array, jnp.array]:
        logits = self.policy_network.apply(self.policy_params, obs)
        value = self.value_network.apply(self.value_params, obs)
        return logits, value

    def ppo_update(self, replay_buffer: ReplayBuffer, ppo_params: PPOParams) -> None:
        obs_buf, act_buf, rew_buf, done_buf, logp_buf, val_buf = replay_buffer.get()

        # Calculate advantages
        adv_buf = rew_buf + (1 - done_buf) * self.gamma * val_buf[1:] - val_buf[:-1]
        adv_buf = (adv_buf - adv_buf.mean()) / (adv_buf.std() + 1e-8)

        # Flatten the batch
        obs_flat = obs_buf.reshape((-1,) + obs_buf.shape[2:])
        act_flat = act_buf.reshape((-1,) + act_buf.shape[2:])
        logp_flat = logp_buf.reshape((-1,) + logp_buf.shape[2:])
        adv_flat = adv_buf.reshape(-1)
        ret_flat = rew_buf.reshape(-1)

        # Update the networks
        for _ in range(ppo_params.num_epochs):
            # Shuffle the indices
            indices = jnp.arange(obs_flat.shape[0])
            indices = jax.random.permutation(jax.random.PRNGKey(0), indices)

            # Iterate over mini-batches
            for start in range(0, obs_flat.shape[0], ppo_params.batch_size):
                end = start + ppo_params.batch_size
                batch_indices = indices[start:end]

                # Policy loss
                logits, _ = self.evaluate(obs_flat[batch_indices])
                ratio = jnp.exp(logits - logp_flat[batch_indices])
                clip_adv = jnp.clip(ratio, 1 - ppo_params.clip_ratio, 1 + ppo_params.clip_ratio) * adv_flat[batch_indices]
                policy_loss = -(jnp.minimum(ratio * adv_flat[batch_indices], clip_adv)).mean()

                # Value loss
                _, v = self.evaluate(obs_flat[batch_indices])
                v_clip = val_buf[batch_indices] + (v - val_buf[batch_indices]).clip(-ppo_params.clip_ratio, ppo_params.clip_ratio)
                v_loss1 = (ret_flat[batch_indices] - v) ** 2
                v_loss2 = (ret_flat[batch_indices] - v_clip) ** 2
                value_loss = 0.5 * jnp.maximum(v_loss1, v_loss2).mean()

                # Entropy bonus
                dist = jax.nn.softmax(logits)
                entropy = -(dist * jnp.log(dist + 1e-8)).sum(-1).mean()

                # Combine losses
                loss = policy_loss + ppo_params.value_coeff * value_loss - ppo_params.entropy_coeff * entropy

                # Update the parameters
                policy_grads = jax.grad(loss, self.policy_params)
                value_grads = jax.grad(loss, self.value_params)
                self.policy_params = self.policy_optimizer.update(policy_grads, self.policy_params)
                self.value_params = self.value_optimizer.update(value_grads, self.value_params)

                # Update the suit distribution predictor
                suit_distribution_loss = self._update_suit_distribution_predictor(obs_flat[batch_indices])
                suit_distribution_grads = jax.grad(suit_distribution_loss, self.suit_distribution_params)
                self.suit_distribution_params = self.suit_distribution_optimizer.update(suit_distribution_grads, self.suit_distribution_params)

    def _update_suit_distribution_predictor(self, obs: jnp.array) -> jnp.array:
        # Implement the loss function and update logic for the suit distribution predictor
        # You can use the observed suit distributions from the game state as the target
        # and calculate the cross-entropy loss between the predicted and target distributions
        ...