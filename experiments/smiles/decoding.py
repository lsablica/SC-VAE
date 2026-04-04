from __future__ import annotations

from typing import Iterable

import torch


def token_ids_to_smiles(
    token_ids: Iterable[int],
    idx_to_token: dict[int, str],
    *,
    stop_at_eos: bool = True,
) -> str:
    decoded = []
    for token_id in token_ids:
        token = idx_to_token[int(token_id)]
        if token == "<eos>" and stop_at_eos:
            break
        if token in {"<pad>", "<sos>"}:
            continue
        decoded.append(token)
    return "".join(decoded)


def logits_to_token_ids(logits: torch.Tensor) -> torch.Tensor:
    if logits.ndim != 3:
        raise ValueError(f"Expected logits with shape [batch, seq, vocab], got {tuple(logits.shape)}")
    return torch.argmax(logits, dim=-1)


def logits_to_smiles_batch(logits: torch.Tensor, idx_to_token: dict[int, str]) -> list[str]:
    token_ids = logits_to_token_ids(logits)
    return [token_ids_to_smiles(row.tolist(), idx_to_token) for row in token_ids]


def deterministic_recon_logits(model, token_batch: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    mu, second_param = model.encode(token_batch)
    logits = model.decode(mu)
    return logits, mu, second_param

