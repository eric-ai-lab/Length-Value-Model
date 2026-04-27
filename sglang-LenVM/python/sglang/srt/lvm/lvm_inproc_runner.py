"""LvmInprocRunner: direct ModelRunner interface for incremental LVM inference.

Bypasses the sglang scheduler and constructs ForwardBatch directly.
Maintains per-request KV state across decode steps so only the 1 new prefix
token (not the full prefix) is forwarded on each step.

Hot path per decode step:
  1. extend_prefix_batch():  1-token extend for each guided request  → O(B)
  2. eval_candidates_batch(): N-candidate tree-attention eval          → O(B*N)

Versus the old approach: O(B * L) tokens forwarded per step.
"""
from __future__ import annotations

import logging
from typing import Dict, List, Optional, TYPE_CHECKING

import torch

from sglang.srt.lvm.tree_value_spec import (
    TreeValueSpecInput,
    build_tree_value_custom_mask_and_positions,
)
from sglang.srt.model_executor.forward_batch_info import (
    ForwardBatch,
    ForwardMode,
    compute_position,
)
from sglang.srt.speculative.spec_info import SpeculativeAlgorithm

if TYPE_CHECKING:
    from sglang.srt.model_executor.model_runner import ModelRunner

logger = logging.getLogger(__name__)


class LvmKvManager:
    """Tracks per-request KV slot lifecycle in the LVM's memory pool.

    Invariant: req_to_token_pool.req_to_token[pool_idx, 0:lvm_kv_lens[rid]]
               holds valid KV slot indices for request rid.
    """

    def __init__(self, runner: "ModelRunner"):
        self.runner = runner
        # rid -> request pool slot index (in req_to_token_pool)
        self.pool_indices: Dict[str, int] = {}
        # rid -> number of prefix tokens whose KV is stored in the pool
        self.kv_lens: Dict[str, int] = {}

    # ------------------------------------------------------------------ #
    # Public API                                                           #
    # ------------------------------------------------------------------ #

    def get_or_alloc(self, rid: str) -> int:
        """Return existing pool index for rid, allocating one on first call."""
        if rid not in self.pool_indices:
            indices = self.runner.req_to_token_pool.alloc(1)
            if indices is None:
                raise RuntimeError(
                    "LVM req_to_token_pool OOM: no free request slots. "
                    "Consider increasing --lvm-guided-inproc-mem-fraction-static."
                )
            self.pool_indices[rid] = indices[0]
            self.kv_lens[rid] = 0
        return self.pool_indices[rid]

    def kv_len(self, rid: str) -> int:
        return self.kv_lens.get(rid, 0)

    def retract(self, rid: str, new_kv_len: int) -> None:
        """Free KV slots beyond new_kv_len (retraction / beam-search rollback)."""
        old_len = self.kv_lens.get(rid, 0)
        if new_kv_len >= old_len:
            return
        pool_idx = self.pool_indices[rid]
        slots = self.runner.req_to_token_pool.req_to_token[
            pool_idx, new_kv_len:old_len
        ].clone()
        self.runner.token_to_kv_pool_allocator.free(slots.to(torch.int64).flatten())
        self.kv_lens[rid] = new_kv_len

    def release(self, rid: str) -> None:
        """Fully release KV and pool slot for a finished / aborted request."""
        if rid not in self.pool_indices:
            return
        pool_idx = self.pool_indices.pop(rid)
        kv_len = self.kv_lens.pop(rid, 0)
        if kv_len > 0:
            slots = self.runner.req_to_token_pool.req_to_token[
                pool_idx, :kv_len
            ].clone()
            self.runner.token_to_kv_pool_allocator.free(
                slots.to(torch.int64).flatten()
            )
        self.runner.req_to_token_pool.free(pool_idx)

    def active_rids(self) -> List[str]:
        return list(self.pool_indices.keys())


class LvmInprocRunner:
    """Direct ModelRunner wrapper for LVM inference without scheduler overhead.

    Two forward pass types per decode step:

    A) extend_prefix_batch - standard EXTEND forward for new prefix tokens
       (usually 1 token per request per step). Writes to LVM KV pool.

    B) eval_candidates_batch - tree-attention EXTEND for N candidates per request.
       Prefix is fully cached (P == L), so Q_len == N (only candidates).
       Candidate KV is allocated temporarily and freed after the forward.
    """

    def __init__(self, runner: "ModelRunner"):
        self.runner = runner
        self.kv_mgr = LvmKvManager(runner)
        self.device = runner.device
        # Use an isolated multimodal embedding cache for LenVM. The decode model and
        # the LenVM may have different hidden sizes, so sharing the global cache can
        # return wrong-shape visual embeddings and silently corrupt the VLM path.
        from sglang.srt.mem_cache.multimodal_cache import MultiModalStaticCache
        import os

        lvm_cache_bytes = int(os.environ.get("SGLANG_LVM_VLM_CACHE_SIZE_MB", 512)) * 1024 * 1024
        self._lvm_embedding_cache = MultiModalStaticCache(lvm_cache_bytes)

    def _lvm_embedding_cache_ctx(self):
        """Temporarily swap in the LenVM-specific multimodal embedding cache."""
        import contextlib
        import sglang.srt.managers.mm_utils as mm_utils_mod

        @contextlib.contextmanager
        def _ctx():
            orig = mm_utils_mod.embedding_cache
            mm_utils_mod.embedding_cache = self._lvm_embedding_cache
            try:
                yield
            finally:
                mm_utils_mod.embedding_cache = orig

        return _ctx()

    # ------------------------------------------------------------------ #
    # Phase A: Extend prefix                                               #
    # ------------------------------------------------------------------ #

    def extend_prefix_batch(
        self,
        rids: List[str],
        new_tokens_list: List[List[int]],
        mm_inputs_list: Optional[List[Optional[object]]] = None,
        mrope_deltas: Optional[dict] = None,
    ) -> None:
        """Extend LVM KV cache for new prefix tokens.

        For each request in rids, new_tokens_list[i] are the tokens that have
        NOT yet been forwarded through the LVM (i.e., the suffix of the full
        prefix that is beyond the current lvm_kv_len).

        mm_inputs_list: optional per-request multimodal inputs for the VLM path.
          - Non-None entries trigger image/video encoding via general_mm_embed_routine.
          - None entries mean text-only extend, but M-RoPE deltas must still be applied.
        mrope_deltas: optional rid -> mrope_position_delta mapping reused after the
          first multimodal extend so later text-only steps stay in the same M-RoPE frame.

        Side-effect: updates lvm_kv_lens and writes KV to the pool.
        No return value (we only need the KV side-effect).
        """
        if not rids:
            return

        runner = self.runner
        device = self.device

        # Allocate/retrieve pool indices.
        pool_indices = [self.kv_mgr.get_or_alloc(rid) for rid in rids]
        extend_lens = [len(toks) for toks in new_tokens_list]
        prefix_lens = [self.kv_mgr.kv_len(rid) for rid in rids]
        seq_lens_list = [prefix_lens[i] + extend_lens[i] for i in range(len(rids))]
        total_extend = sum(extend_lens)

        if total_extend == 0:
            return

        # Allocate KV slots for the new tokens.
        allocator = runner.token_to_kv_pool_allocator
        if getattr(allocator, "page_size", 1) == 1:
            out_cache_loc = allocator.alloc(total_extend)
        else:
            last_loc = torch.empty(len(pool_indices), dtype=torch.int64, device=device)
            for i, (pool_idx, p_len) in enumerate(zip(pool_indices, prefix_lens)):
                last_loc[i] = (
                    runner.req_to_token_pool.req_to_token[pool_idx, p_len - 1]
                    if p_len > 0
                    else -1
                )
            out_cache_loc = allocator.alloc_extend(
                torch.tensor(prefix_lens, dtype=torch.int64, device=device),
                torch.tensor(prefix_lens, dtype=torch.int64),
                torch.tensor(seq_lens_list, dtype=torch.int64, device=device),
                torch.tensor(seq_lens_list, dtype=torch.int64),
                last_loc,
                total_extend,
            )
        if out_cache_loc is None:
            raise RuntimeError(
                f"LVM KV pool OOM: cannot allocate {total_extend} slots for "
                "extend_prefix_batch. Consider --lvm-guided-inproc-mem-fraction-static."
            )
        out_cache_loc = out_cache_loc.to(torch.int64)

        # Write new slots into req_to_token_pool (existing prefix slots are
        # already there from previous calls – we only write the new portion).
        pt = 0
        for pool_idx, p_len, e_len in zip(pool_indices, prefix_lens, extend_lens):
            runner.req_to_token_pool.write(
                (pool_idx, slice(p_len, p_len + e_len)),
                out_cache_loc[pt : pt + e_len],
            )
            pt += e_len

        # Build ForwardBatch tensors.
        input_ids_t = torch.tensor(
            [t for toks in new_tokens_list for t in toks],
            dtype=torch.int64,
            device=device,
        )
        req_pool_indices_t = torch.tensor(pool_indices, dtype=torch.int64, device=device)
        seq_lens_t = torch.tensor(seq_lens_list, dtype=torch.int32, device=device)
        seq_lens_cpu_t = torch.tensor(seq_lens_list, dtype=torch.int32)
        extend_prefix_lens_t = torch.tensor(prefix_lens, dtype=torch.int32, device=device)
        extend_seq_lens_t = torch.tensor(extend_lens, dtype=torch.int32, device=device)

        positions, extend_start_loc = compute_position(
            runner.server_args.attention_backend,
            extend_prefix_lens_t,
            extend_seq_lens_t,
            total_extend,
        )

        forward_batch = ForwardBatch(
            forward_mode=ForwardMode.EXTEND,
            batch_size=len(rids),
            input_ids=input_ids_t,
            req_pool_indices=req_pool_indices_t,
            seq_lens=seq_lens_t,
            seq_lens_cpu=seq_lens_cpu_t,
            out_cache_loc=out_cache_loc,
            seq_lens_sum=sum(seq_lens_list),
            positions=positions,
            extend_num_tokens=total_extend,
            extend_seq_lens=extend_seq_lens_t,
            extend_prefix_lens=extend_prefix_lens_t,
            extend_start_loc=extend_start_loc,
            extend_prefix_lens_cpu=prefix_lens,
            extend_seq_lens_cpu=extend_lens,
            req_to_token_pool=runner.req_to_token_pool,
            token_to_kv_pool=runner.token_to_kv_pool,
            attn_backend=runner.attn_backend,
            return_logprob=False,
            is_extend_in_batch=True,
            is_prefill_only=True,
            spec_algorithm=SpeculativeAlgorithm.NONE,
            global_forward_mode=ForwardMode.EXTEND,
        )
        forward_batch.num_token_non_padded_cpu = total_extend

        if mm_inputs_list is not None:
            forward_batch.mm_inputs = list(mm_inputs_list)
            mrope_chunks = []
            for i, (mm_inp, p_len, e_len) in enumerate(
                zip(mm_inputs_list, prefix_lens, extend_lens)
            ):
                if mm_inp is not None:
                    mrope_pos = getattr(mm_inp, "mrope_positions", None)
                    if mrope_pos is not None:
                        chunk = mrope_pos[:, p_len : p_len + e_len].to(device)
                    else:
                        delta = getattr(mm_inp, "mrope_position_delta", None)
                        offset = int(delta.item()) if delta is not None else 0
                        pos_range = torch.arange(
                            p_len, p_len + e_len, device=device, dtype=torch.int64
                        ) + offset
                        chunk = pos_range.unsqueeze(0).expand(3, -1)
                else:
                    delta_t = mrope_deltas.get(rids[i]) if mrope_deltas else None
                    offset = int(delta_t.item()) if delta_t is not None else 0
                    pos_range = torch.arange(
                        p_len, p_len + e_len, device=device, dtype=torch.int64
                    ) + offset
                    chunk = pos_range.unsqueeze(0).expand(3, -1)
                mrope_chunks.append(chunk)
            if mrope_chunks:
                forward_batch.mrope_positions = torch.cat(mrope_chunks, dim=1)

            for mm_input in mm_inputs_list:
                if mm_input is not None and hasattr(mm_input, "mm_items"):
                    for mm_item in mm_input.mm_items:
                        feature = getattr(mm_item, "feature", None)
                        if isinstance(feature, torch.Tensor) and not feature.is_cuda:
                            mm_item.feature = feature.to(device)

        with self._lvm_embedding_cache_ctx():
            runner.forward_extend(forward_batch)

        # Update kv_lens (KV for new tokens is now in the pool).
        for rid, e_len in zip(rids, extend_lens):
            self.kv_mgr.kv_lens[rid] += e_len

    # ------------------------------------------------------------------ #
    # Phase B: Evaluate candidates                                         #
    # ------------------------------------------------------------------ #

    def eval_candidates_batch_gpu(
        self,
        rids: List[str],
        candidate_ids_per_req: List[List[int]],
        gpu_candidates: Optional[tuple] = None,
        mrope_deltas: Optional[dict] = None,
    ) -> List[torch.Tensor]:
        """Evaluate candidates with tree attention and return raw GPU embeddings.

        mrope_deltas: optional rid -> mrope_position_delta mapping used by the
        VLM path to build 3D M-RoPE positions for candidate tokens.

        This enables the fast-path where LenVM mathematical guidance runs entirely
        on the GPU without downloading tensors to the CPU.
        """
        if not rids:
            return []

        runner = self.runner
        device = self.device

        prefix_lens = [self.kv_mgr.kv_len(rid) for rid in rids]
        cand_lens = [len(cands) for cands in candidate_ids_per_req]
        seq_lens_list = [prefix_lens[i] + cand_lens[i] for i in range(len(rids))]
        total_cands = sum(cand_lens)
        pool_indices = [self.kv_mgr.pool_indices[rid] for rid in rids]

        # Allocate temporary KV slots for candidate tokens.
        allocator = runner.token_to_kv_pool_allocator
        if getattr(allocator, "page_size", 1) == 1:
            out_cache_loc = allocator.alloc(total_cands)
        else:
            last_loc = torch.empty(len(pool_indices), dtype=torch.int64, device=device)
            for i, (pool_idx, p_len) in enumerate(zip(pool_indices, prefix_lens)):
                last_loc[i] = runner.req_to_token_pool.req_to_token[pool_idx, p_len - 1]
            out_cache_loc = allocator.alloc_extend(
                torch.tensor(prefix_lens, dtype=torch.int64, device=device),
                torch.tensor(prefix_lens, dtype=torch.int64),
                torch.tensor(seq_lens_list, dtype=torch.int64, device=device),
                torch.tensor(seq_lens_list, dtype=torch.int64),
                last_loc,
                total_cands,
            )
        if out_cache_loc is None:
            raise RuntimeError(
                f"LVM KV pool OOM: cannot allocate {total_cands} candidate slots. "
                "Consider --lvm-guided-inproc-mem-fraction-static."
            )
        out_cache_loc = out_cache_loc.to(torch.int64)

        # Write candidate slot indices into req_to_token_pool (after prefix).
        pt = 0
        for pool_idx, p_len, n in zip(pool_indices, prefix_lens, cand_lens):
            runner.req_to_token_pool.write(
                (pool_idx, slice(p_len, p_len + n)),
                out_cache_loc[pt : pt + n],
            )
            pt += n

        # Build input tensors (only candidate tokens — prefix is all cached).
        if gpu_candidates is not None:
            _, gi, gm = gpu_candidates
            input_ids_t = gi[gm].to(torch.int64)
        else:
            input_ids_t = torch.tensor(
                [t for cands in candidate_ids_per_req for t in cands],
                dtype=torch.int64,
                device=device,
            )
        
        req_pool_indices_t = torch.tensor(pool_indices, dtype=torch.int64, device=device)
        seq_lens_t = torch.tensor(seq_lens_list, dtype=torch.int32, device=device)
        seq_lens_cpu_t = torch.tensor(seq_lens_list, dtype=torch.int32)
        extend_prefix_lens_t = torch.tensor(prefix_lens, dtype=torch.int32, device=device)
        extend_seq_lens_t = torch.tensor(cand_lens, dtype=torch.int32, device=device)

        # Tree attention mask: P == L, so Q_len == N (candidates only).
        # Each candidate attends to [0..L-1] (prefix in KV) and itself.
        custom_mask, positions = build_tree_value_custom_mask_and_positions(
            prefix_lens=prefix_lens,
            candidate_lens=cand_lens,
            cached_prefix_lens=prefix_lens,  # P == L: full cache hit
            device=device,
        )
        spec_info = TreeValueSpecInput(
            custom_mask=custom_mask,
            positions=positions,
            tree_value_prefix_lens=list(prefix_lens),
            tree_value_candidate_lens=list(cand_lens),
            tree_value_cached_prefix_lens=list(prefix_lens),
        )

        # extend_start_loc: cumulative sum of extend_seq_lens (= cand_lens).
        extend_start_loc = torch.zeros(len(rids), dtype=torch.int32, device=device)
        if len(rids) > 1:
            extend_start_loc[1:] = torch.cumsum(extend_seq_lens_t[:-1], dim=0)

        forward_batch = ForwardBatch(
            forward_mode=ForwardMode.EXTEND,
            batch_size=len(rids),
            input_ids=input_ids_t,
            req_pool_indices=req_pool_indices_t,
            seq_lens=seq_lens_t,
            seq_lens_cpu=seq_lens_cpu_t,
            out_cache_loc=out_cache_loc,
            seq_lens_sum=sum(seq_lens_list),
            positions=positions,  # overridden by spec_info.positions in the model
            extend_num_tokens=total_cands,
            extend_seq_lens=extend_seq_lens_t,
            extend_prefix_lens=extend_prefix_lens_t,
            extend_start_loc=extend_start_loc,
            extend_prefix_lens_cpu=list(prefix_lens),
            extend_seq_lens_cpu=list(cand_lens),
            req_to_token_pool=runner.req_to_token_pool,
            token_to_kv_pool=runner.token_to_kv_pool,
            attn_backend=runner.attn_backend,
            return_logprob=False,
            is_extend_in_batch=True,
            is_prefill_only=True,
            spec_algorithm=SpeculativeAlgorithm.NONE,
            spec_info=spec_info,
            global_forward_mode=ForwardMode.EXTEND,
        )
        forward_batch.num_token_non_padded_cpu = total_cands

        if mrope_deltas is not None:
            cand_mrope_chunks = []
            for rid, p_len, n in zip(rids, prefix_lens, cand_lens):
                delta = mrope_deltas.get(rid)
                offset = int(delta.item()) if delta is not None else 0
                pos = p_len + offset
                cand_mrope_chunks.append(
                    torch.full((3, n), pos, dtype=torch.int64, device=device)
                )
            forward_batch.mrope_positions = torch.cat(cand_mrope_chunks, dim=1)

        try:
            logits_output = runner.forward_extend(forward_batch)
        finally:
            # Free candidate KV slots immediately (they must not be cached).
            runner.token_to_kv_pool_allocator.free(out_cache_loc)

        return logits_output.embeddings

    def eval_candidates_batch(
        self,
        rids: List[str],
        candidate_ids_per_req: List[List[int]],
    ) -> List[List[float]]:
        """Evaluate candidates with tree attention.

        Assumes extend_prefix_batch has been called so that
        lvm_kv_lens[rid] == len(prefix_ids) for each rid.

        The prefix is fully cached (P == L), so the custom mask only has
        N candidate rows per request (Q_len == N), which is much cheaper
        than the old approach where Q_len == (L-P) + N ≈ L + N.

        Candidate KV slots are allocated temporarily and freed after forward.

        Returns: List[List[float]] of raw (pre-sigmoid) logit values per candidate.
        """
        embeddings = self.eval_candidates_batch_gpu(rids, candidate_ids_per_req)
        out: List[List[float]] = []
        for t in embeddings:
            out.append(t.detach().cpu().float().tolist())
        return out

    # ------------------------------------------------------------------ #
    # Lifecycle helpers                                                    #
    # ------------------------------------------------------------------ #

    def cleanup_stale_rids(self, active_rids: set) -> None:
        """Release KV for requests no longer in the running batch."""
        stale = [
            rid
            for rid in self.kv_mgr.active_rids()
            if rid not in active_rids
        ]
        for rid in stale:
            self.kv_mgr.release(rid)
