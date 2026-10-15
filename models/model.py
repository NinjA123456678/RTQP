from __future__ import annotations

import math

import torch
from torch import nn

from .config import ExperimentConfig, ablation_flags


class SDAE(nn.Module):
    def __init__(self, config: ExperimentConfig):
        super().__init__()
        dims = [config.window] + [max(4, config.sdae_width // (2 ** i)) for i in range(config.stacks)]
        self.masking = config.masking
        encoders, decoders = [], []
        for a, b in zip(dims[:-1], dims[1:]):
            encoders.extend([nn.Linear(a, b), nn.ReLU()])
        for i in range(len(dims) - 1, 0, -1):
            decoders.extend([nn.Linear(dims[i], dims[i - 1]), nn.ReLU()])
        self.encoder, self.decoder = nn.Sequential(*encoders), nn.Sequential(*decoders)

    def forward(self, x, observed=None):
        if self.training:
            dropped = torch.rand_like(x) < self.masking
            if observed is not None:
                dropped &= observed
            x = x.masked_fill(dropped, 0)
        return self.decoder(self.encoder(x))


def reliability(x, reconstruction, observed):
    """Eqs. 5--7: min/max within each available causal window, excluding missing entries."""
    error = (x - reconstruction).square()
    low = error.masked_fill(~observed, float("inf")).amin(dim=-1, keepdim=True)
    high = error.masked_fill(~observed, -float("inf")).amax(dim=-1, keepdim=True)
    span = high - low
    score = torch.where(span > 1e-8, (error - low) / span.clamp_min(1e-8), torch.zeros_like(error))
    score = torch.nan_to_num(score).clamp(0, 1)
    confidence = (1 - score) * observed.to(x.dtype)
    return score, confidence


class TemporalEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dim, self.heads = 2 * config.channels, config.heads
        self.convs = nn.ModuleList([nn.Conv1d(1, config.channels, k, padding=k // 2) for k in (3, 5)])
        self.qkv = nn.Linear(self.dim, 3 * self.dim)
        self.out = nn.Linear(self.dim, self.dim)

    def forward(self, x, confidence, observed, attention=True, weighted=True):
        h = torch.cat([torch.relu(conv(x[:, None])) for conv in self.convs], dim=1).transpose(1, 2)
        if not attention:
            weights = observed.to(h.dtype).unsqueeze(-1)
            return (h * weights).sum(1) / weights.sum(1).clamp_min(1), None
        b, t, _ = h.shape
        q, k, v = self.qkv(h).chunk(3, dim=-1)
        def heads(value):
            return value.reshape(b, t, self.heads, self.dim // self.heads).transpose(1, 2)
        q, k, v = heads(q), heads(k), heads(v)
        logits = q @ k.transpose(-2, -1) / math.sqrt(self.dim // self.heads)
        eligible = observed[:, None, None, :]
        # Eq. 17 is softmax(e) * w / sum(softmax(e) * w), with EXACT zeros.
        alpha = torch.softmax(logits.masked_fill(~eligible, -1e9), dim=-1) * eligible
        weights = alpha * confidence[:, None, None, :] if weighted else alpha
        total = weights.sum(-1, keepdim=True)
        plain_total = alpha.sum(-1, keepdim=True)
        fallback = alpha / torch.where(plain_total > 0, plain_total, torch.ones_like(plain_total))
        weights = torch.where(total > 0,
                              weights / torch.where(total > 0, total, torch.ones_like(total)), fallback)
        context = (weights @ v).transpose(1, 2).reshape(b, t, self.dim)
        # Last historical query predicts the next slice (Eqs. 18 and 24).
        feature = self.out(context[:, -1])
        feature = feature * observed.any(-1, keepdim=True)
        return feature, weights


class FeatureEnhancement(nn.Module):
    """Eq. 20 interpreted as four source tokens, avoiding length-one attention."""
    def __init__(self, config):
        super().__init__()
        self.rank = config.rank
        self.attention = nn.MultiheadAttention(config.rank, config.heads, batch_first=True)
        width = 4 * config.rank
        self.convs = nn.Sequential(nn.Conv1d(config.rank, width, 3, padding=1), nn.ReLU(),
                                   nn.Conv1d(width, width, 3, padding=1), nn.ReLU(),
                                   nn.Conv1d(width, width, 3, padding=1), nn.ReLU())
        self.norm = nn.LayerNorm(4 * config.rank)

    def forward(self, tokens, enhanced_branch=True):
        active = 4 if enhanced_branch else 2
        h = tokens[:, :active]
        # Explicit attention weights avoid nondeterministic fused SDPA backends.
        h, _ = self.attention(h, h, h, need_weights=True)
        h = self.convs(h.transpose(1, 2)).mean(dim=-1)  # Average pooling over source tokens.
        return self.norm(h)


class Predictor(nn.Module):
    def __init__(self, input_dim, hidden, dropout):
        super().__init__()
        self.fc1, self.fc2 = nn.Linear(input_dim, hidden), nn.Linear(hidden, hidden)
        self.dropout, self.norm = nn.Dropout(dropout), nn.LayerNorm(hidden)
        self.output = nn.Linear(hidden, 1)

    def forward(self, feature):
        first = torch.relu(self.fc1(feature))
        second = self.fc2(first)
        # Eq. 27: first FFN state + Dropout(second FFN state).
        return self.output(self.norm(first + self.dropout(second))).squeeze(-1)


class RTQP(nn.Module):
    """Stage-2 network with a frozen SDAE and cached latest-slice NMF features."""
    def __init__(self, config: ExperimentConfig, sdae_state=None):
        super().__init__()
        self.flags = ablation_flags(config.case)
        self.sdae = SDAE(config) if self.flags["reliability"] else None
        if self.sdae is not None:
            if sdae_state is not None:
                self.sdae.load_state_dict(sdae_state)
            self.sdae.requires_grad_(False).eval()
        self.temporal = TemporalEncoder(config)
        self.enhancer = FeatureEnhancement(config)
        self.predictor = Predictor(2 * config.channels + 4 * config.rank, config.hidden, config.dropout)

    def train(self, mode=True):
        super().train(mode)
        if self.sdae is not None:
            self.sdae.eval()  # Stage 2 never enables SDAE random masking.
        return self

    def forward(self, batch, return_attention=False):
        f = self.flags
        original = batch["x"]
        if self.sdae is not None:
            with torch.no_grad():
                reconstructed = self.sdae(original, batch["observed"])
                _, confidence = reliability(original, reconstructed, batch["observed"])
                enhanced = confidence * original + (1 - confidence) * reconstructed
                enhanced = enhanced * batch["observed"]
        else:
            confidence, enhanced = batch["observed"].to(original.dtype), original
        x = enhanced if f["reconstruction_fusion"] else original
        temporal, attention = self.temporal(x, confidence, batch["observed"],
                                            f["attention"], f["confidence_attention"])
        tokens = batch["nmf"]
        if not f["enhanced_branch"]:
            tokens = torch.cat([tokens[:, :2], torch.zeros_like(tokens[:, 2:])], dim=1)
        if f["feature_enhancement"]:
            interaction = self.enhancer(tokens, f["enhanced_branch"])
        else:
            interaction = tokens.flatten(1)  # Case 3 retains ALL four NMF sources.
        prediction = self.predictor(torch.cat([interaction, temporal], dim=-1))
        return (prediction, attention) if return_attention else prediction

    def loss(self, batch):
        error = (self(batch) - batch["target"]).square()
        if self.flags["confidence_loss"]:
            error = error * batch["target_confidence"]
        return error.mean()  # Eq. 28 divides by |Omega|, NOT sum(confidence).
