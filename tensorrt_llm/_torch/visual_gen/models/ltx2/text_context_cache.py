# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

"""LTX2 text context cache — caches constant text-derived computations.

Two storage strategies:

* ``_SlicedBuffer``: fixed ``[2,...]`` buffer with per-CFG-pass batch slicing.
  Used for tensors that differ between COND and UNCOND (context, mask, KV).
* ``_SimpleBuffer``: plain store/return without batch manipulation.
  Used for position-only tensors identical across CFG passes (PE, cross-PE).

CFG modes (for ``_SlicedBuffer``):
    COND   → buf[1:2]   (index 1)
    UNCOND → buf[0:1]   (index 0)
    BOTH   → buf[:]     (full, BasePipeline [uncond, cond] concat)
"""

from __future__ import annotations

from enum import IntEnum
from typing import Optional

import torch

# Batch indices matching BasePipeline concat order [neg, pos].
_UNCOND_IDX = 0
_COND_IDX = 1


class CfgPass(IntEnum):
    COND = 0
    UNCOND = 1
    BOTH = 2


class ModalityType(IntEnum):
    VIDEO = 0
    AUDIO = 1


def _cfg_slice(cfg_pass: CfgPass) -> slice:
    if cfg_pass == CfgPass.COND:
        return slice(_COND_IDX, _COND_IDX + 1)
    if cfg_pass == CfgPass.UNCOND:
        return slice(_UNCOND_IDX, _UNCOND_IDX + 1)
    return slice(None)


def _expand2(t: torch.Tensor) -> torch.Tensor:
    """Repeat a batch=1 tensor to batch=2."""
    return t.repeat(2, *([1] * (t.dim() - 1)))


class _SlicedBuffer:
    """Fixed [2,...] buffer with per-CFG-pass batch slicing.

    Used for tensors whose content differs between COND and UNCOND
    (context embeddings, attention masks, KV projections).
    """

    __slots__ = ("_buf",)

    def __init__(self) -> None:
        self._buf: Optional[torch.Tensor] = None

    def store(self, cfg_pass: CfgPass, tensor: torch.Tensor) -> None:
        s = _cfg_slice(cfg_pass)
        if self._buf is not None and self._buf.shape[1:] == tensor.shape[1:]:
            self._buf[s].copy_(tensor)
        else:
            self._buf = _expand2(tensor) if tensor.shape[0] < 2 else tensor.clone()

    def get(self, cfg_pass: CfgPass) -> Optional[torch.Tensor]:
        if self._buf is None:
            return None
        return self._buf[_cfg_slice(cfg_pass)]


class _SimpleBuffer:
    """Plain store/return buffer — no batch expansion or slicing.

    Used for position-only tensors (PE, cross-PE) that are identical for
    COND and UNCOND and rely on batch=1 broadcasting.
    """

    __slots__ = ("_buf",)

    def __init__(self) -> None:
        self._buf: Optional[torch.Tensor] = None

    def store(self, tensor: torch.Tensor) -> None:
        if self._buf is not None and self._buf.shape == tensor.shape:
            self._buf.copy_(tensor)
        else:
            self._buf = tensor.clone()

    def get(self) -> Optional[torch.Tensor]:
        return self._buf


class LTX2TextContextCache:
    """Text context cache with mixed buffer strategies.

    Args:
        num_layers: Number of transformer blocks.
    """

    def __init__(self, num_layers: int) -> None:
        self.num_layers = num_layers
        n = len(ModalityType)

        # CFG-dependent: context, mask → _SlicedBuffer
        self._ctx = [_SlicedBuffer() for _ in range(n)]
        self._mask = [_SlicedBuffer() for _ in range(n)]

        # Position-only: PE, cross-PE → _SimpleBuffer (no CFG slicing)
        self._pe = [(_SimpleBuffer(), _SimpleBuffer()) for _ in range(n)]
        self._cross_pe = [(_SimpleBuffer(), _SimpleBuffer()) for _ in range(n)]

        # [modality][cfg_pass] → dirty
        self._preproc_dirty: list[list[bool]] = [[True, True] for _ in range(n)]

        # KV buffers: CFG-dependent → _SlicedBuffer
        self._kv = [
            [(_SlicedBuffer(), _SlicedBuffer()) for _ in range(num_layers)] for _ in range(n)
        ]
        self._kv_dirty: list[bool] = [True, True]  # [cfg_pass]

    # -- Lifecycle ---------------------------------------------------------

    def invalidate(self) -> None:
        """Mark all dirty.  Clear cross-PE (resolution may change across stages)."""
        for m in range(len(ModalityType)):
            self._preproc_dirty[m] = [True, True]
            # cross_pe has no dirty flag — clear buffers so stale data from a
            # previous stage/resolution is not returned by get_cross_pe().
            self._cross_pe[m] = (_SimpleBuffer(), _SimpleBuffer())
        self._kv_dirty = [True, True]

    # -- Preprocessor ------------------------------------------------------

    def store_preproc(
        self,
        cfg_pass: CfgPass,
        modality: ModalityType,
        context: torch.Tensor,
        mask: Optional[torch.Tensor],
        pe: tuple[torch.Tensor, torch.Tensor],
    ) -> None:
        self._ctx[modality].store(cfg_pass, context)
        if mask is not None:
            self._mask[modality].store(cfg_pass, mask)
        # PE is position-only — simple store, no CFG slicing.
        self._pe[modality][0].store(pe[0])
        self._pe[modality][1].store(pe[1])
        if cfg_pass == CfgPass.BOTH:
            self._preproc_dirty[modality] = [False, False]
        else:
            self._preproc_dirty[modality][cfg_pass] = False

    def get_preproc(
        self, cfg_pass: CfgPass, modality: ModalityType
    ) -> Optional[tuple[torch.Tensor, Optional[torch.Tensor], tuple[torch.Tensor, torch.Tensor]]]:
        """Return ``(context, mask, (pe_cos, pe_sin))`` or ``None`` if dirty."""
        d = self._preproc_dirty[modality]
        if cfg_pass == CfgPass.BOTH:
            if d[0] or d[1]:
                return None
        elif d[cfg_pass]:
            return None
        return (
            self._ctx[modality].get(cfg_pass),
            self._mask[modality].get(cfg_pass),
            (self._pe[modality][0].get(), self._pe[modality][1].get()),
        )

    # -- cross_pe ----------------------------------------------------------

    def store_cross_pe(
        self,
        cfg_pass: CfgPass,
        modality: ModalityType,
        cross_pe: tuple[torch.Tensor, torch.Tensor],
    ) -> None:
        self._cross_pe[modality][0].store(cross_pe[0])
        self._cross_pe[modality][1].store(cross_pe[1])

    def get_cross_pe(
        self, cfg_pass: CfgPass, modality: ModalityType
    ) -> Optional[tuple[torch.Tensor, torch.Tensor]]:
        c = self._cross_pe[modality][0].get()
        if c is None:
            return None
        return (c, self._cross_pe[modality][1].get())

    # -- KV ----------------------------------------------------------------

    def kv_is_dirty(self, cfg_pass: CfgPass = CfgPass.COND) -> bool:
        if cfg_pass == CfgPass.BOTH:
            return self._kv_dirty[0] or self._kv_dirty[1]
        return self._kv_dirty[cfg_pass]

    def kv_mark_clean(self, cfg_pass: CfgPass = CfgPass.COND) -> None:
        if cfg_pass == CfgPass.BOTH:
            self._kv_dirty = [False, False]
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
        self._kv[modality][layer_idx][0].store(cfg_pass, k)
        self._kv[modality][layer_idx][1].store(cfg_pass, v)

    def get_kv(
        self, modality: ModalityType, cfg_pass: CfgPass, layer_idx: int
    ) -> tuple[torch.Tensor, torch.Tensor]:
        kb, vb = self._kv[modality][layer_idx]
        return kb.get(cfg_pass), vb.get(cfg_pass)
