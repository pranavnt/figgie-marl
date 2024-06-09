import jax.numpy as jnp
from flax import struct

@struct.dataclass
class PPOParams:
    clip_ratio: float
    entropy_coeff: float
    value_coeff: float
    max_grad_norm: float
    num_epochs: int
    num_mini_batches: int

class ReplayBuffer:
    def __init__(self, obs_dim: int, act_dim: int, capacity: int) -> None:
        self.obs_buf = jnp.zeros((capacity, obs_dim))
        self.act_buf = jnp.zeros((capacity, act_dim))
        self.rew_buf = jnp.zeros(capacity)
        self.done_buf = jnp.zeros(capacity, dtype=bool)
        self.logp_buf = jnp.zeros(capacity)
        self.val_buf = jnp.zeros(capacity)
        self.capacity = capacity
        self.ptr, self.size = 0, 0

    def store(self, obs: jnp.array, act: jnp.array, rew: float, done: bool, logp: float, val: float) -> None:
        self.obs_buf = self.obs_buf.at[self.ptr].set(obs)
        self.act_buf = self.act_buf.at[self.ptr].set(act)
        self.rew_buf = self.rew_buf.at[self.ptr].set(rew)
        self.done_buf = self.done_buf.at[self.ptr].set(done)
        self.logp_buf = self.logp_buf.at[self.ptr].set(logp)
        self.val_buf = self.val_buf.at[self.ptr].set(val)
        self.ptr = (self.ptr + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)