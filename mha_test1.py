from typing import Optional
import equinox as eqx
import jax
import jax.numpy as jnp
from icecream import ic
from jaxtyping import Array, Float, Bool, PRNGKeyArray, PyTree
import math
import functools as ft
import time

# ic.disable()

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


def get_positional_encoding(
    n_tokens: int, n_vocab: int
) -> Float[Array, "n_tokens n_vocab"]:
    pos = jnp.arange(n_tokens)[:, jnp.newaxis]
    div_term = jnp.exp(jnp.arange(0, n_vocab, 2) * -(jnp.log(10000.0) / n_vocab))
    # alternatively: div_term = 1 / 10000 ** (jnp.arange(0, D, 2) / D)
    # that's closer to the actual notation they used.
    pos_enc = jnp.zeros((n_tokens, n_vocab))
    pos_enc = pos_enc.at[:, 0::2].set(jnp.sin(pos * div_term))
    pos_enc = pos_enc.at[:, 1::2].set(jnp.cos(pos * div_term))
    return pos_enc


def dot_product_attention(
    query_projection: Float[Array, "max_seq_len query_embedding_dim"],
    key_projection: Float[Array, "max_seq_len key_embedding_dim"],
    value_projection: Float[Array, "max_seq_len value_embedding_dim"],
    mask: Optional[Array | None | Bool] = None,
) -> Array:
    attention_weights = query_projection @ key_projection.T
    attention_weights = attention_weights / jnp.sqrt(key_projection.shape[-1])
    if mask is not None:
        T = attention_weights.shape[-1]
        mask = jnp.tril(jnp.ones(shape=(T, T))) == 1
        # ic(mask.shape)
        # mask = jnp.expand_dims(mask, axis=1) # we add an extra dimension at axis 1 for broadcasting
        # ic(mask.shape)
        attention_weights = jnp.where(mask, attention_weights, float("-inf"))
    attention_weights = jax.nn.softmax(attention_weights, axis=-1)
    qkv_matmul = attention_weights @ value_projection
    return qkv_matmul


def vmapped_attention(query_heads, key_heads, value_heads, mask):
    attn_fn = ft.partial(dot_product_attention, mask=mask)
    # Inner VMAP
    dpa = jax.vmap(
        lambda q, k, v: attn_fn(q, k, v),
        in_axes=(1, None, None),
        out_axes=1,
    )(query_heads, key_heads, value_heads)
    return dpa


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

    kv_multihead_dim: Optional[int | None] = eqx.field(static=True)

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
        query_multihead_dim: int,
        kv_multihead_dim: Optional[int],
        key,
    ):
        qkey, kkey, vkey, okey = jax.random.split(key, 4)
        self.query_projection = eqx.nn.Linear(
            query_input_dim, num_heads * query_embedding_dim, key=qkey, use_bias=False
        )
        self.key_projection = eqx.nn.Linear(
            key_input_dim,
            kv_multihead_dim * key_embedding_dim,
            key=kkey,
            use_bias=False,
        )
        self.value_projection = eqx.nn.Linear(
            value_input_dim,
            kv_multihead_dim * value_embedding_dim,
            key=vkey,
            use_bias=False,
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

    def __call__(self, x: Float[Array, "max_seq_len input_dim"], masking: bool = False):
        seq_len, _ = x.shape
        query = jax.vmap(self.query_projection)(x).reshape(
            seq_len, self.num_heads, self.query_embedding_dim
        )
        key = jax.vmap(self.key_projection)(x).reshape(
            seq_len, self.kv_multihead_dim, self.key_embedding_dim
        )
        value = jax.vmap(self.value_projection)(x).reshape(
            seq_len, self.kv_multihead_dim, self.value_embedding_dim
        )

        pt_vmapped_fn = ft.partial(
            vmapped_attention,
            mask=masking,
        )

        # Outer VMAP
        qkv_matmul = jax.vmap(
            pt_vmapped_fn,
            in_axes=(None, 1, 1),
        )(query, key, value)

        qkv_matmul = jnp.sum(qkv_matmul, axis=0)

        # Taking the mean over the d dimension
        qkv_matmul = qkv_matmul / self.kv_multihead_dim
        concatenation = qkv_matmul.reshape(seq_len, -1)
        output = jax.vmap(self.output)(concatenation)
        return output


class RMSNorm(eqx.Module):
    weight: Array
    eps: float

    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = jnp.ones(dim)

    def _norm(self, x: Array):
        return x * jax.lax.rsqrt(jnp.mean(x**2, axis=-1, keepdims=True) + self.eps)

    def __call__(self, x: Array) -> Array:
        output = self._norm(x)
        return output * self.weight


class Transformer(eqx.Module):
    input_embedding: eqx.nn.Embedding
    masked_mha: MultiheadAttention
    feedforward: eqx.nn.MLP
    rms_norm: RMSNorm

    output: eqx.nn.Linear
    positional_encoding: Array

    def __init__(
        self,
        n_dims: int,
        n_embd: int,
        n_heads: int,
        key: PRNGKeyArray,
        width_size: int = 32,
        depth: int = 2,
        max_token_size: int = 8,
    ) -> None:
        key, *subkeys = jax.random.split(
            key, 20
        )  # let's just split 20 for now, we'll probably need them later
        self.input_embedding = eqx.nn.Embedding(n_dims, n_embd, key=subkeys[0])
        self.masked_mha = MultiheadAttention(
            query_input_dim=n_embd,
            key_input_dim=n_embd,
            value_input_dim=n_embd,
            query_embedding_dim=n_embd // n_heads,
            key_embedding_dim=n_embd // (n_heads),
            value_embedding_dim=n_embd // (n_heads),
            num_heads=n_heads,
            output_dim=n_embd,
            query_multihead_dim=n_heads,
            kv_multihead_dim=n_heads - 2,
            key=subkeys[1],
        )

        # Equinox has a built-in MLP module
        self.feedforward = eqx.nn.MLP(
            in_size=n_embd,
            out_size=n_embd,
            width_size=width_size,
            key=subkeys[2],
            depth=depth,
        )
        self.positional_encoding = get_positional_encoding(max_token_size, n_embd)

        self.rms_norm = RMSNorm(dim=n_embd)

        self.output = eqx.nn.Linear(
            in_features=n_embd, out_features=n_dims, key=subkeys[4], use_bias=False
        )

    def __call__(self, x):
        x = jax.vmap(self.input_embedding)(x)
        x += self.positional_encoding
        x = self.rms_norm(self.masked_mha(x, masking=True) + x)  # residual connection
        x = self.rms_norm(jax.vmap(self.feedforward)(x) + x)  # residual connection
        x = jax.vmap(self.output)(x)
        # x = jax.nn.softmax(x) # we don't softmax here, because we want the raw logits for our loss function
        # but you can totally softmax here and inverse that later;
        return x


def main():
    from tinyshakespeareloader.hamlet import get_data
    import optax

    data = get_data()

    train_dataloader, test_dataloader, vocabulary_size, chars, encode, decode = (
        data["train_dataloader"],
        data["test_dataloader"],
        data["vocabulary_size"],
        data["chars"],
        data["encode"],
        data["decode"],
    )
    key = jax.random.PRNGKey(420)
    INPUT_DIMS: int = int(vocabulary_size)
    N_EMBD = 32
    N_HEADS = 4
    MAX_T = 8

    def loss_fn(transformer: Transformer, x: Array, y: Array):
        logits = eqx.filter_vmap(transformer)(x)
        loss = optax.softmax_cross_entropy_with_integer_labels(logits, y)

        return jnp.mean(loss)

    def evaluate(transformer: Transformer, test_dataloader):
        loss = 0
        jitted_loss_fn = eqx.filter_jit(loss_fn)
        for x, y in test_dataloader:
            x = jnp.array(x.numpy())
            y = jnp.array(y.numpy())
            loss += jitted_loss_fn(transformer, x, y)

        return loss / len(test_dataloader)

    @eqx.filter_jit
    def step(
        transformer: PyTree,
        opt_state: optax.OptState,
        optimiser: optax.GradientTransformation,
        x: Array,
        y: Array,
    ):
        loss, grads = eqx.filter_value_and_grad(loss_fn)(transformer, x, y)
        updates, opt_state = optimiser.update(grads, opt_state, transformer)
        transformer = eqx.apply_updates(transformer, updates)
        return transformer, opt_state, loss

    transformer = Transformer(
        n_dims=INPUT_DIMS, n_embd=N_EMBD, n_heads=N_HEADS, key=key
    )
    # start_loss = evaluate(transformer, test_dataloader)
    # print(f"{start_loss=}")
    optimiser = optax.adamw(learning_rate=0.001)
    opt_state = optimiser.init(eqx.filter(transformer, eqx.is_inexact_array))
    ic("starting training")
    start_time = time.time()
    for i, (x, y) in enumerate(train_dataloader):
        x = jnp.array(x.numpy())
        y = jnp.array(y.numpy())
        transformer, opt_state, loss = step(transformer, opt_state, optimiser, x, y)
        if i % 100 == 0:
            eval_loss = evaluate(transformer, test_dataloader)
            ic(i, loss, eval_loss)
        if i == 1000:
            ic("early stopping")
            break
    end_time = time.time()
    ic("done training")
    ic("training time:", end_time - start_time)
    ic(evaluate(transformer, test_dataloader))


if __name__ == "__main__":
    ic.enable()
    main()
