from typing import Optional
import jax
import jax.numpy as jnp
import jax.random as jr
import equinox as eqx

from custom_types import PRNGKeyArray, XArray, Array, Float, Scalar, typecheck

QArray = Float[Array, "..."]

AArray = Float[Array, "..."]


"""
    Diffusion Transformer (DiT)
"""


class AdaLayerNorm(eqx.Module):
    norm: eqx.nn.LayerNorm
    scale_proj: eqx.nn.Linear
    shift_proj: eqx.nn.Linear

    @typecheck
    def __init__(self, embed_dim: int, *, key: PRNGKeyArray):
        keys = jr.split(key)
        self.norm = eqx.nn.LayerNorm(embed_dim)
        self.scale_proj = eqx.nn.Linear(embed_dim, embed_dim, key=keys[0])
        self.shift_proj = eqx.nn.Linear(embed_dim, embed_dim, key=keys[1])

    @typecheck
    def __call__(self, x: Float[Array, "q"], y: Float[Array, "y"]):
        gamma = self.scale_proj(y)
        beta = self.shift_proj(y) 
        return self.norm(x) * (1. + gamma) + beta


class PatchEmbedding(eqx.Module):
    patch_size: int
    proj: eqx.nn.Conv2d
    cls_token: Float[Array, "1 1 e"]
    pos_embed: Float[Array, "1 s e"]

    @typecheck
    def __init__(
        self, 
        img_size: int, 
        patch_size: int, 
        in_channels: int, 
        embed_dim: int, 
        key: PRNGKeyArray
    ):
        keys = jr.split(key, 3)
        self.patch_size = patch_size
        self.proj = eqx.nn.Conv2d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size, key=keys[0])
        self.cls_token = jr.normal(keys[1], (1, embed_dim)) # extra 1 before
        self.pos_embed = jr.normal(keys[2], (int(img_size / patch_size) ** 2 + 1, embed_dim))

    @typecheck
    def __call__(self, x: Float[Array, "_ _ _"]) -> Float[Array, "s q"]:
        x = self.proj(x)
        x = rearrange(x, "c h w -> (h w) c") 
        x = jnp.concatenate([self.cls_token, x], axis=0) 
        x = x + self.pos_embed  
        return x


class TimestepEmbedding(eqx.Module):
    embed_dim: int
    mlp: eqx.nn.Sequential

    @typecheck
    def __init__(self, embed_dim: int, *, key: PRNGKeyArray):
        self.embed_dim = embed_dim

        keys = jr.split(key)
        self.mlp = eqx.nn.Sequential(
            [
                eqx.nn.Linear(1, embed_dim, key=keys[0]),
                eqx.nn.Lambda(jax.nn.gelu),
                eqx.nn.Linear(embed_dim, embed_dim, key=keys[1]),
            ]
        )

    @typecheck
    def __call__(self, t: Scalar) -> Float[Array, "{self.embed_dim}"]:
        return self.mlp(jnp.atleast_1d(t))


class TransformerBlock(eqx.Module):
    norm1: AdaLayerNorm
    attn: eqx.nn.MultiheadAttention
    norm2: AdaLayerNorm
    mlp: eqx.nn.Sequential

    @typecheck
    def __init__(
        self, 
        embed_dim: int, 
        n_heads: int, 
        *, 
        key: PRNGKeyArray
    ):
        keys = jr.split(key, 5)
        self.norm1 = AdaLayerNorm(embed_dim, key=keys[0])
        self.attn = eqx.nn.MultiheadAttention(n_heads, embed_dim, key=keys[1]) # NOTE: Casting in here...
        self.norm2 = AdaLayerNorm(embed_dim, key=keys[2])
        self.mlp = eqx.nn.Sequential(
            [
                eqx.nn.Linear(embed_dim, embed_dim * 4, key=keys[3]),
                eqx.nn.Lambda(jax.nn.gelu),
                eqx.nn.Linear(embed_dim * 4, embed_dim, key=keys[4])
            ]
        )

    @typecheck
    def __call__(self, x: Float[Array, "s q"], y: Float[Array, "y"]) -> Float[Array, "s q"]:
        x = precision_cast(jax.vmap(lambda x: self.norm1(x, y)), x)
        x = x + self.attn(x, x, x)
        x = precision_cast(jax.vmap(lambda x: self.norm2(x, y)), x)
        x = x + jax.vmap(self.mlp)(x)
        return x


class DiT(eqx.Module):
    img_size: int
    q_dim: int
    patch_embed: PatchEmbedding
    time_embed: TimestepEmbedding
    a_embed: Optional[eqx.nn.Linear]
    blocks: list[TransformerBlock]
    out_conv: eqx.nn.ConvTranspose2d
    scaler: Optional[eqx.Module] = None

    @typecheck
    def __init__(
        self, 
        img_size: int, 
        patch_size: int, 
        channels: int, 
        embed_dim: int, 
        depth: int, 
        n_heads: int, 
        q_dim: Optional[int] = None, 
        a_dim: Optional[int] = None, 
        scaler: Optional[eqx.Module] = None,
        *, 
        key: PRNGKeyArray
    ):
        self.img_size = img_size
        self.q_dim = q_dim

        keys = jr.split(key, 5)
        channels = channels + q_dim if (q_dim is not None) else channels

        self.patch_embed = PatchEmbedding(
            img_size, patch_size, channels, embed_dim, key=keys[0]
        )
        self.time_embed = TimestepEmbedding(embed_dim, key=keys[1])
        
        self.a_embed = eqx.nn.Linear(
            a_dim, embed_dim, key=keys[2]
        ) if (a_dim is not None) else None

        block_keys = jr.split(keys[3], depth)
        self.blocks = eqx.filter_vmap(
            lambda key: TransformerBlock(embed_dim, n_heads, key=key) 
        )(block_keys)

        self.out_conv = eqx.nn.ConvTranspose2d(
            embed_dim, 
            channels, 
            kernel_size=patch_size, 
            stride=patch_size, 
            key=keys[4]
        )

        self.scaler = scaler

    @typecheck
    def __call__(
        self, 
        t: Scalar, 
        x: XArray, 
        q: QArray, 
        a: AArray,
        key: Optional[PRNGKeyArray] = None
    ) -> Float[Array, "_ _ _"]:

        if exists(self.scaler):
            x, q, a = self.scaler.forward(x, q, a)

        x = self.patch_embed(
            jnp.concatenate([x, q]) 
            if (q is not None) and (self.q_dim is not None)
            else x
        )

        t_embedding = self.time_embed(t)
        if (a is not None) and (self.a_dim is not None):
            a_embedding = self.a_embed(a)
            embedding = a_embedding + t_embedding
        else:
            embedding = t_embedding

        all_params, struct = eqx.partition(self.blocks, eqx.is_array)

        def block_fn(x, params):
            block = eqx.combine(params, struct)
            x = block(x, embedding)
            return x, None

        x, _ = jax.lax.scan(block_fn, x, all_params)

        x = x[1:] # No class token 

        x = rearrange(
            x, 
            "(h w) c -> c h w", 
            h=int(self.img_size / self.patch_embed.patch_size)
        )  

        x = self.out_conv(
            jnp.concatenate([x, q]) 
            if (q is not None) and (self.q_dim is not None)
            else x
        )

        return x