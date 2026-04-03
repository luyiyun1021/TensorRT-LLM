# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

"""LTX2 text context cache — caches constant text-derived computations across denoise steps.

Text context (prompt embeddings) is constant throughout the denoising loop.
This module caches two levels of derived computation:

1. **Preprocessor outputs** (per modality): ``caption_projection(context)``,
   attention mask, RoPE positional embeddings, and cross-PE.
2. **Per-block KV projections** (per layer per modality): ``to_k(context)``,
   ``to_v(context)``, and ``norm_k(k)`` for text cross-attention.

Supports 2 CFG slots (conditional + unconditional) so that single-GPU CFG
does not pollute the cache.

Lifecycle:
- Created once by the pipeline (survives across requests).
- ``invalidate()`` marks all slots dirty before each denoising loop.
  Buffers are retained for reuse via ``copy_()``.
- ``LTXModel.forward()`` computes KV projections and stores via ``store_kv()``.
- Compiled blocks read via ``get_kv()``.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum

import torch


class CfgSlot(IntEnum):
    """CFG pass type — used as cache slot index."""

    COND = 0
    UNCOND = 1


class ModalityType(IntEnum):
    """Modality type — used as cache dimension index."""

    VIDEO = 0
    AUDIO = 1


@dataclass
class _PreprocEntry:
    """Cached preprocessor outputs for one modality in one CFG slot."""

    context: torch.Tensor | None = None
    mask: torch.Tensor | None = None
    pe: tuple[torch.Tensor, torch.Tensor] | None = None
    cross_pe: tuple[torch.Tensor, torch.Tensor] | None = None


class LTX2TextContextCache:
    """Caches text-derived computations across denoise steps.

    Args:
        num_layers: Number of transformer blocks.
        max_batch_size: Maximum batch size for pre-allocated KV buffers.
    """

    def __init__(self, num_layers: int, max_batch_size: int = 1) -> None:
        self.num_layers = num_layers
        self.max_batch_size = max_batch_size

        num_slots = len(CfgSlot)
        num_modalities = len(ModalityType)

        # Preprocessor cache: [slot][modality] → _PreprocEntry
        self._preproc: list[list[_PreprocEntry]] = [
            [_PreprocEntry() for _ in range(num_modalities)] for _ in range(num_slots)
        ]

        # Per-block KV cache: [slot][modality][layer] → (k, v) | None
        # Buffers are allocated on first fill and retained across invalidate().
        self._kv: list[list[list[tuple[torch.Tensor, torch.Tensor] | None]]] = [
            [[None] * num_layers for _ in range(num_modalities)] for _ in range(num_slots)
        ]
        # Dirty flag per slot — True means needs refill.
        self._kv_dirty: list[bool] = [True] * num_slots

    def invalidate(self) -> None:
        """Mark all slots dirty.  Buffers are retained for ``copy_()`` reuse."""
        for s in CfgSlot:
            for m in ModalityType:
                self._preproc[s][m] = _PreprocEntry()
            self._kv_dirty[s] = True

    # -- Preprocessor cache ------------------------------------------------

    def get_preproc(self, is_unconditional: bool, modality: ModalityType) -> _PreprocEntry:
        """Return the preprocessor cache entry for reading/writing."""
        return self._preproc[CfgSlot.UNCOND if is_unconditional else CfgSlot.COND][modality]

    # -- KV cache ----------------------------------------------------------

    def kv_is_dirty(self, is_unconditional: bool) -> bool:
        """Return True if this slot needs KV refill."""
        return self._kv_dirty[CfgSlot.UNCOND if is_unconditional else CfgSlot.COND]

    def kv_mark_clean(self, is_unconditional: bool) -> None:
        """Mark this slot as filled.  Call after storing all layers."""
        self._kv_dirty[CfgSlot.UNCOND if is_unconditional else CfgSlot.COND] = False

    def store_kv(
        self,
        modality: ModalityType,
        is_unconditional: bool,
        layer_idx: int,
        k: torch.Tensor,
        v: torch.Tensor,
    ) -> None:
        """Store KV for one layer.  Reuses buffer on subsequent calls."""
        s = CfgSlot.UNCOND if is_unconditional else CfgSlot.COND
        existing = self._kv[s][modality][layer_idx]
        if existing is not None:
            existing[0].copy_(k)
            existing[1].copy_(v)
        else:
            self._kv[s][modality][layer_idx] = (k.clone(), v.clone())

    def get_kv(
        self, modality: ModalityType, is_unconditional: bool, layer_idx: int
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return cached KV for a compiled block."""
        s = CfgSlot.UNCOND if is_unconditional else CfgSlot.COND
        return self._kv[s][modality][layer_idx]
