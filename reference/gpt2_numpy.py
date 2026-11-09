import json
import math
from pathlib import Path

import numpy as np
from safetensors.numpy import load_file


def layer_norm(x, weight, bias, epsilon):
    mean = x.mean(axis=-1, keepdims=True)
    variance = ((x - mean) ** 2).mean(axis=-1, keepdims=True)
    return (x - mean) / np.sqrt(variance + np.float32(epsilon)) * weight + bias


def gelu(x):
    coefficient = np.float32(math.sqrt(2.0 / math.pi))
    return (
        np.float32(0.5)
        * x
        * (np.float32(1.0) + np.tanh(coefficient * (x + np.float32(0.044715) * x**3)))
    )


class GPT2:
    def __init__(self, model_dir):
        model_dir = Path(model_dir)
        config = json.loads((model_dir / "config.json").read_text())
        raw = load_file(str(model_dir / "model.safetensors"))

        weights = {}
        for name, value in raw.items():
            if name.startswith("transformer."):
                name = name[len("transformer.") :]
            if name.endswith(".attn.bias") or name.endswith(".attn.masked_bias"):
                continue
            weights[name] = np.ascontiguousarray(value, dtype=np.float32)

        self.n_layer = config["n_layer"]
        self.n_head = config["n_head"]
        self.n_embd = config["n_embd"]
        self.n_ctx = config["n_ctx"]
        self.head_dim = self.n_embd // self.n_head
        self.epsilon = config["layer_norm_epsilon"]
        self.scale = np.float32(1.0 / math.sqrt(self.head_dim))

        self.wte = weights["wte.weight"]
        self.wpe = weights["wpe.weight"]
        self.ln_f_weight = weights["ln_f.weight"]
        self.ln_f_bias = weights["ln_f.bias"]

        self.layers = []
        for index in range(self.n_layer):
            prefix = f"h.{index}."
            self.layers.append(
                {
                    "ln_1_weight": weights[prefix + "ln_1.weight"],
                    "ln_1_bias": weights[prefix + "ln_1.bias"],
                    "attn_c_attn_weight": weights[prefix + "attn.c_attn.weight"],
                    "attn_c_attn_bias": weights[prefix + "attn.c_attn.bias"],
                    "attn_c_proj_weight": weights[prefix + "attn.c_proj.weight"],
                    "attn_c_proj_bias": weights[prefix + "attn.c_proj.bias"],
                    "ln_2_weight": weights[prefix + "ln_2.weight"],
                    "ln_2_bias": weights[prefix + "ln_2.bias"],
                    "mlp_c_fc_weight": weights[prefix + "mlp.c_fc.weight"],
                    "mlp_c_fc_bias": weights[prefix + "mlp.c_fc.bias"],
                    "mlp_c_proj_weight": weights[prefix + "mlp.c_proj.weight"],
                    "mlp_c_proj_bias": weights[prefix + "mlp.c_proj.bias"],
                }
            )

    def _split_heads(self, x):
        return x.reshape(x.shape[0], self.n_head, self.head_dim).transpose(1, 0, 2)

    def _attention(self, x, layer, cache):
        qkv = x @ layer["attn_c_attn_weight"] + layer["attn_c_attn_bias"]
        query, key, value = np.split(qkv, 3, axis=-1)
        query = self._split_heads(query)
        key = self._split_heads(key)
        value = self._split_heads(value)

        if cache is not None:
            key = np.concatenate((cache[0], key), axis=1)
            value = np.concatenate((cache[1], value), axis=1)

        query_length = query.shape[1]
        key_length = key.shape[1]
        scores = query @ key.transpose(0, 2, 1) * self.scale

        query_positions = np.arange(key_length - query_length, key_length)
        key_positions = np.arange(key_length)
        masked = key_positions[None, :] > query_positions[:, None]
        scores = np.where(masked[None, :, :], np.float32(-np.inf), scores)

        scores = scores - scores.max(axis=-1, keepdims=True)
        probabilities = np.exp(scores)
        probabilities /= probabilities.sum(axis=-1, keepdims=True)

        context = probabilities @ value
        context = context.transpose(1, 0, 2).reshape(query_length, self.n_embd)
        attention = context @ layer["attn_c_proj_weight"] + layer["attn_c_proj_bias"]
        return attention, key, value

    def _mlp(self, x, layer):
        hidden = x @ layer["mlp_c_fc_weight"] + layer["mlp_c_fc_bias"]
        hidden = gelu(hidden)
        return hidden @ layer["mlp_c_proj_weight"] + layer["mlp_c_proj_bias"]

    def _block(self, x, layer, cache):
        normalized = layer_norm(x, layer["ln_1_weight"], layer["ln_1_bias"], self.epsilon)
        attention, key, value = self._attention(normalized, layer, cache)
        x = x + attention

        normalized = layer_norm(x, layer["ln_2_weight"], layer["ln_2_bias"], self.epsilon)
        x = x + self._mlp(normalized, layer)
        return x, key, value

    def _logits(self, x):
        x = layer_norm(x, self.ln_f_weight, self.ln_f_bias, self.epsilon)
        return x @ self.wte.T

    def prefill(self, ids):
        if not ids:
            raise ValueError("prompt must contain at least one token")
        x = self.wte[np.asarray(ids, dtype=np.int64)] + self.wpe[: len(ids)]
        caches = []
        for layer in self.layers:
            x, key, value = self._block(x, layer, None)
            caches.append((key, value))
        return self._logits(x[-1]), caches

    def step(self, token, position, caches):
        x = (self.wte[token] + self.wpe[position])[None, :]
        new_caches = []
        for layer, cache in zip(self.layers, caches):
            x, key, value = self._block(x, layer, cache)
            new_caches.append((key, value))
        return self._logits(x[0]), new_caches
