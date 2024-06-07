from env import FiggieEnv
import jax
from jax import jit, grad
import jax.numpy as jnp
from jax import random
import jax.nn as nn

def policy_network(obs, params):
    logits = jnp.dot(obs, params['w']) + params['b']
    return nn.softmax(logits)

def value_network(obs, params):
    return jnp.dot(obs, params['w']) + params['b']

@jit
def policy_loss(params, obs, action, advantage):
    probs = policy_network(obs, params)
    log_prob = jnp.log(probs[action[0]])
    return -log_prob * advantage

@jit
def value_loss(params, obs, return_):
    value_pred = value_network(obs, params)
    return jnp.square(value_pred - return_)[0]

@jit
def update_step(params, grads, learning_rate=0.01):
    params['w'] -= learning_rate * grads['w']
    params['b'] -= learning_rate * grads['b']
    return params

env = FiggieEnv(num_players=4)

key = random.PRNGKey(0)
num_actions = 4
action_suit_range = 4
params = {
    'w': random.normal(key, (4, num_actions * action_suit_range)),
    'b': random.normal(key, (num_actions * action_suit_range,))
}

policy_params = {
    'w': random.normal(key, (4, num_actions * action_suit_range)),
    'b': random.normal(key, (num_actions * action_suit_range,))
}
value_params = {
    'w': random.normal(key, (4, 1)),
    'b': random.normal(key, (1,))
}

# Training loop
num_episodes = 1000
for episode in range(num_episodes):
    obs = env.reset()
    done = False
    while not done:
        action_probs = policy_network(obs['player_cards'], policy_params)
        action_type = jax.random.categorical(random.PRNGKey(episode), action_probs[:num_actions])
        suit = jax.random.categorical(random.PRNGKey(episode), action_probs[num_actions:num_actions*2])
        amount = jax.random.randint(random.PRNGKey(episode), (1,), 1, 41)[0]  # Amount from 1 to 40

        action = (int(action_type), int(suit), int(amount))
        next_obs, reward, done, _ = env.step(action)

        value = value_network(obs['player_cards'], value_params)[0]
        next_value = value_network(next_obs['player_cards'], value_params)[0]
        td_error = reward + (1 - done) * next_value - value
        return_ = reward + (1 - done) * next_value

        policy_grads = grad(policy_loss)(policy_params, obs['player_cards'], action, td_error)

        value_grads = grad(value_loss)(value_params, obs['player_cards'], return_)
        policy_params = update_step(policy_params, policy_grads)
        value_params = update_step(value_params, value_grads)

        obs = next_obs
