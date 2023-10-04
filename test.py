from typing import Optional

import equinox as eqx
import jax
import jax.numpy as jnp
from icecream import ic
from jaxtyping import Array, Float, Bool
import math
import functools as ft


query_input_dim = 16
query_embedding_dim = 32
key_input_dim = 16
key_embedding_dim = 32
value_input_dim = 16
value_embedding_dim = 32
num_heads = 4
max_seq_len = 10
batch_size = 2
output_dim = 32
kv_multihead_dim = 4
query_multihead_dim = 4


key = jax.random.PRNGKey(42)


def dot_product_attention(
    query_projection: Array,
    key_projection: Array,
    value_projection: Array,
    mask: Optional[Array | None],
) -> Array:
    ic(query_projection.shape, key_projection.shape, value_projection.shape)

    attention_weights = query_projection @ key_projection.T
    attention_weights = attention_weights / jnp.sqrt(key_projection.shape[-1])
    attention_weights = jax.nn.softmax(attention_weights, axis=-1)
    ic(attention_weights.shape)

    qkv_matmul = attention_weights @ value_projection

    return qkv_matmul


def vmapped_attention(query_heads, key_heads, value_heads, mask):
    attn_fn = ft.partial(dot_product_attention, mask=mask)
    ic(query_heads.shape, key_heads.shape, value_heads.shape)
    dpa = jax.vmap(
        lambda q, k, v: attn_fn(q, k, v),
        in_axes=(1, None, None),
        out_axes=1,
    )(query_heads, key_heads, value_heads)
    return dpa


# Version 1
class MultiheadAttention(eqx.Module):
    query_projection: eqx.nn.Linear
    key_projection: eqx.nn.Linear
    value_projection: eqx.nn.Linear

    query_input_dim: int = eqx.field(static=True)
    query_embedding_dim: int = eqx.field(static=True)
    query_multihead_dim: int = eqx.field(static=True)

    key_input_dim: int = eqx.field(static=True)
    key_embedding_dim: int = eqx.field(static=True)

    value_input_dim: int = eqx.field(static=True)
    value_embedding_dim: int = eqx.field(static=True)

    kv_multihead_dim: int = eqx.field(static=True)

    output: eqx.nn.Linear
    num_heads: int = eqx.field(static=True)
    output_dim: int = eqx.field(static=True)

    def __init__(
        self,
        query_embedding_dim,
        key_embedding_dim,
        value_embedding_dim,
        query_input_dim,
        key_input_dim,
        value_input_dim,
        num_heads,
        output_dim,
        query_multihead_dim,
        kv_multihead_dim,
        key,
    ):
        qkey, kkey, vkey, okey = jax.random.split(key, 4)
        self.query_projection = eqx.nn.Linear(
            query_input_dim, num_heads * query_embedding_dim, key=qkey, use_bias=False
        )
        self.key_projection = eqx.nn.Linear(
            key_input_dim, num_heads * key_embedding_dim, key=kkey, use_bias=False
        )
        self.value_projection = eqx.nn.Linear(
            value_input_dim, num_heads * value_embedding_dim, key=vkey, use_bias=False
        )

        self.output = eqx.nn.Linear(
            num_heads * value_embedding_dim, output_dim, key=okey, use_bias=False
        )

        # parameters
        self.query_input_dim = query_input_dim
        self.query_embedding_dim = query_embedding_dim
        self.key_input_dim = key_input_dim
        self.key_embedding_dim = key_embedding_dim
        self.value_input_dim = value_input_dim
        self.value_embedding_dim = value_embedding_dim
        self.num_heads = num_heads
        self.output_dim = output_dim
        self.query_multihead_dim = query_multihead_dim
        self.kv_multihead_dim = kv_multihead_dim

    def __call__(self, x: Float[Array, "max_seq_len input_dim"]):
        seq_len, _ = x.shape
        query = jax.vmap(self.query_projection)(x).reshape(
            seq_len, self.num_heads, self.query_embedding_dim
        )
        key = jax.vmap(self.key_projection)(x).reshape(
            seq_len, self.num_heads, self.key_embedding_dim
        )
        value = jax.vmap(self.value_projection)(x).reshape(
            seq_len, self.num_heads, self.value_embedding_dim
        )

        pt_vmapped_fn = ft.partial(
            vmapped_attention,
            mask=None,
        )

        ic(query.shape, key.shape, value.shape)

        qkv_matmul = jax.vmap(
            pt_vmapped_fn,
            in_axes=(None, 1, 1),
        )(query, key, value)

        qkv_matmul = jnp.sum(qkv_matmul, axis=0)

        # Taking the mean over the d dimension
        qkv_matmul = qkv_matmul / self.kv_multihead_dim
        ic(qkv_matmul.shape)
        concatenation = qkv_matmul.reshape(seq_len, -1)

        ic(concatenation.shape)

        output = jax.vmap(self.output)(concatenation)
        ic(output.shape)
        return output


key, subkey = jax.random.split(key)
mha = MultiheadAttention(
    query_embedding_dim,
    key_embedding_dim,
    value_embedding_dim,
    query_input_dim,
    key_input_dim,
    value_input_dim,
    num_heads,
    output_dim,
    query_embedding_dim,
    kv_multihead_dim,
    key,
)
x = jax.random.normal(subkey, (max_seq_len, query_input_dim))
output = mha(x)
