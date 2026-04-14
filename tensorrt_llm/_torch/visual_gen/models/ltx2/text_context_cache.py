# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

"""LTX2 text context cache — fixed [2,...] shared buffer design.

All buffers have batch=2 regardless of CFG mode.  COND/UNCOND write to
their respective slice; BOTH writes the full buffer.  No grow/shrink.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum
from typing import Optional

import torch

# Batch indices: BasePipeline concats [neg, pos] → 0=UNCOND, 1=COND.
_UNCOND_IDX = 0
_COND_IDX = 1


class CfgPass(IntEnum):
    """CFG pass type."""

    COND = 0
    UNCOND = 1
    BOTH = 2


class ModalityType(IntEnum):
    """Modality type."""

    VIDEO = 0
    AUDIO = 1


def _cfg_slice(cfg_pass: CfgPass) -> slice:
    """Return batch slice for a CFG pass."""
    if cfg_pass == CfgPass.COND:
        return slice(_COND_IDX, _COND_IDX + 1)
    if cfg_pass == CfgPass.UNCOND:
        return slice(_UNCOND_IDX, _UNCOND_IDX + 1)
    return slice(None)  # BOTH


@dataclass
class _PreprocEntry:
    """Preprocessor cache entry.  All tensors have batch=2 when filled."""

    context: Optional[torch.Tensor] = None
    mask: Optional[torch.Tensor] = None
    pe: Optional[tuple[torch.Tensor, torch.Tensor]] = None
    cross_pe: Optional[tuple[torch.Tensor, torch.Tensor]] = None

    @property
    def is_filled(self) -> bool:
        return self.context is not None

    def get(self, s: slice) -> _PreprocEntry:
        """Return sliced view."""
        if not self.is_filled:
            return _PreprocEntry()

        def _sl(t):
            return t[s] if t is not None else None

        def _sl2(p):
            return (p[0][s], p[1][s]) if p is not None else None

        return _PreprocEntry(
            context=_sl(self.context),
            mask=_sl(self.mask),
            pe=_sl2(self.pe),
            cross_pe=_sl2(self.cross_pe),
        )

    def copy_slice(self, s: slice, src: _PreprocEntry) -> None:
        """Copy src into this entry's slice in-place."""
        if src.context is not None and self.context is not None:
            self.context[s].copy_(src.context)
        if src.mask is not None and self.mask is not None:
            self.mask[s].copy_(src.mask)
        if src.pe is not None and self.pe is not None:
            self.pe[0][s].copy_(src.pe[0])
            self.pe[1][s].copy_(src.pe[1])
        # cross_pe handled separately by MultiModal preprocessor


def _expand2(t: torch.Tensor) -> torch.Tensor:
    """Repeat batch=1 tensor to batch=2."""
    return t.repeat(2, *([1] * (t.dim() - 1)))


def _alloc_batch2(src: _PreprocEntry) -> _PreprocEntry:
    """Allocate batch=2 entry from a source (batch=1 or batch=2)."""
    if src.context.shape[0] == 2:
        return _PreprocEntry(
            context=src.context.clone(),
            mask=src.mask.clone() if src.mask is not None else None,
            pe=(src.pe[0].clone(), src.pe[1].clone()) if src.pe is not None else None,
        )
    return _PreprocEntry(
        context=_expand2(src.context),
        mask=_expand2(src.mask) if src.mask is not None else None,
        pe=(_expand2(src.pe[0]), _expand2(src.pe[1])) if src.pe is not None else None,
    )


class LTX2TextContextCache:
    """Fixed [2,...] shared buffer cache.

    Args:
        num_layers: Number of transformer blocks.
    """

    def __init__(self, num_layers: int) -> None:
        self.num_layers = num_layers
        n_mod = len(ModalityType)

        self._preproc: list[_PreprocEntry] = [_PreprocEntry() for _ in range(n_mod)]
        self._preproc_dirty: list[bool] = [True, True]  # [COND, UNCOND]

        self._kv: list[list[Optional[tuple[torch.Tensor, torch.Tensor]]]] = [
            [None] * num_layers for _ in range(n_mod)
        ]
        self._kv_dirty: list[bool] = [True, True]

    def invalidate(self) -> None:
        """Mark all dirty.  Buffers retained."""
        for m in ModalityType:
            self._preproc[m] = _PreprocEntry()
        self._preproc_dirty = [True, True]
        self._kv_dirty = [True, True]

    # -- Preprocessor ------------------------------------------------------

    def get_preproc(self, cfg_pass: CfgPass, modality: ModalityType) -> _PreprocEntry:
        """Return sliced view of preprocessor entry.  Empty if slot is dirty."""
        if cfg_pass == CfgPass.BOTH:
            if self._preproc_dirty[CfgPass.COND] or self._preproc_dirty[CfgPass.UNCOND]:
                return _PreprocEntry()
        elif self._preproc_dirty[cfg_pass]:
            return _PreprocEntry()
        entry = self._preproc[modality]
        if not entry.is_filled:
            return _PreprocEntry()
        return entry.get(_cfg_slice(cfg_pass))

    def store_preproc(self, cfg_pass: CfgPass, modality: ModalityType, src: _PreprocEntry) -> None:
        """Store preprocessor outputs into shared [2,...] buffer."""
        entry = self._preproc[modality]
        s = _cfg_slice(cfg_pass)

        if entry.is_filled and entry.context.shape[1:] == src.context.shape[1:]:
            entry.copy_slice(s, src)
        else:
            new_entry = _alloc_batch2(src)
            new_entry.copy_slice(s, src)
            self._preproc[modality] = new_entry

        if cfg_pass == CfgPass.BOTH:
            self._preproc_dirty[CfgPass.COND] = False
            self._preproc_dirty[CfgPass.UNCOND] = False
        else:
            self._preproc_dirty[cfg_pass] = False

    # -- KV ----------------------------------------------------------------

    def kv_is_dirty(self, cfg_pass: CfgPass = CfgPass.COND) -> bool:
        if cfg_pass == CfgPass.BOTH:
            return self._kv_dirty[CfgPass.COND] or self._kv_dirty[CfgPass.UNCOND]
        return self._kv_dirty[cfg_pass]

    def kv_mark_clean(self, cfg_pass: CfgPass = CfgPass.COND) -> None:
        if cfg_pass == CfgPass.BOTH:
            self._kv_dirty[CfgPass.COND] = False
            self._kv_dirty[CfgPass.UNCOND] = False
        else:
            self._kv_dirty[cfg_pass] = False

    def store_kv(
        self,
        modality: ModalityType,
        cfg_pass: CfgPass,
        layer_idx: int,
        k: torch.Tensor,
        v: torch.Tensor,
    ) -> None:
        """Store KV into shared [2,...] buffer."""
        s = _cfg_slice(cfg_pass)
        existing = self._kv[modality][layer_idx]

        if existing is not None and existing[0].shape[1:] == k.shape[1:]:
            existing[0][s].copy_(k)
            existing[1][s].copy_(v)
        else:
            if k.shape[0] == 2:
                self._kv[modality][layer_idx] = (k.clone(), v.clone())
            else:
                buf_k = _expand2(k)
                buf_v = _expand2(v)
                buf_k[s].copy_(k)
                buf_v[s].copy_(v)
                self._kv[modality][layer_idx] = (buf_k, buf_v)

    def get_kv(
        self, modality: ModalityType, cfg_pass: CfgPass, layer_idx: int
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return sliced KV view."""
        k, v = self._kv[modality][layer_idx]
        s = _cfg_slice(cfg_pass)
        return k[s], v[s]
