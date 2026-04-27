from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

import torch

from sglang.srt.speculative.spec_info import SpecInput, SpecInputType


@dataclass
class TreeValueSpecInput(SpecInput):
    """SpecInput-like container to reuse custom tree attention mask plumbing.

    This is NOT speculative decoding. We only leverage:
      - `custom_mask` (flattened per-seq QxK masks)
      - `positions` (flattened positions for extend tokens)

    And carry extra metadata for LVM head slicing:
      - `tree_value_prefix_lens`: List[int]
      - `tree_value_candidate_lens`: List[int]

    Mask layout matches triton backend expectations:
      - `custom_mask` is a flattened concatenation of per-seq masks
      - each per-seq mask is row-major [Q_len, K_len] where:
          Q_len = extend_seq_len = (L - P) + N
          K_len = seq_len = P + extend_seq_len = L + N
      - `mask_indptr` is computed in attention backend as cumulative sum of (Q_len * K_len).
    """

    custom_mask: torch.Tensor
    positions: torch.Tensor
    tree_value_prefix_lens: List[int]
    tree_value_candidate_lens: List[int]
    tree_value_cached_prefix_lens: List[int]

    def __post_init__(self):
        super().__init__(SpecInputType.EAGLE_VERIFY)

    def get_spec_adjust_token_coefficient(self) -> Tuple[int, int]:
        # No token multiplier for DP buffers.
        return 1, 1

    def _per_req_qk_and_offsets(self) -> Tuple[List[int], List[int], List[int], List[int]]:
        """
        Compute per-request:
          q_len = (L - P) + N
          k_len = L + N
          mask_len = q_len * k_len
        And cumulative offsets into flattened tensors:
          mask_offsets[i] = start offset for request i in custom_mask
          pos_offsets[i]  = start offset for request i in positions
        """
        Ls = [int(x) for x in self.tree_value_prefix_lens]
        Ns = [int(x) for x in self.tree_value_candidate_lens]
        Ps = [int(x) for x in self.tree_value_cached_prefix_lens]
        if not (len(Ls) == len(Ns) == len(Ps)):
            raise ValueError("TreeValueSpecInput lens length mismatch")

        q_lens: List[int] = []
        k_lens: List[int] = []
        mask_lens: List[int] = []
        mask_offsets: List[int] = []
        pos_offsets: List[int] = []
        m_off = 0
        p_off = 0
        for L, N, P in zip(Ls, Ns, Ps):
            q = (int(L) - int(P)) + int(N)
            k = int(L) + int(N)
            if q <= 0 or k <= 0:
                raise ValueError(f"Invalid q/k computed for tree_value: L={L} N={N} P={P}")
            q_lens.append(q)
            k_lens.append(k)
            ml = q * k
            mask_lens.append(ml)
            mask_offsets.append(m_off)
            pos_offsets.append(p_off)
            m_off += ml
            p_off += q
        return q_lens, k_lens, mask_offsets, pos_offsets

    def filter_batch(self, new_indices: torch.Tensor, has_been_filtered: bool = True):
        """
        Keep only a subset of the batch in the order specified by `new_indices`.
        This is required because ScheduleBatch.filter_batch() unconditionally calls
        spec_info.filter_batch() when spec_info exists.

        Notes:
        - `new_indices` is a 1D int64 tensor on device.
        - `has_been_filtered=True` means the batch has already been filtered once elsewhere
          (e.g., by unfinished_index), and scheduler is calling filter again for bookkeeping.
          In that case, some spec implementations simply truncate to the new batch size.
        """
        if new_indices is None:
            return
        if new_indices.numel() == 0:
            # Empty batch
            self.custom_mask = self.custom_mask[:0]
            self.positions = self.positions[:0]
            self.tree_value_prefix_lens = []
            self.tree_value_candidate_lens = []
            self.tree_value_cached_prefix_lens = []
            return

        idx_list = new_indices.to("cpu", non_blocking=True).tolist()

        if has_been_filtered:
            # In some code paths (e.g., spec v1 verify), the batch may have already been filtered
            # elsewhere. In that *common* case, `new_indices` is simply [0..new_bs) and we can
            # safely truncate for speed.
            #
            # However, under heavy load / reordering, we must still handle non-trivial index sets
            # correctly; otherwise custom_mask/positions can become misaligned with the requests,
            # causing invalid attention reads and NaN/Inf.
            new_bs = len(idx_list)
            if idx_list == list(range(new_bs)):
                if new_bs >= len(self.tree_value_prefix_lens):
                    return
                q_lens, _k_lens, mask_offsets, pos_offsets = self._per_req_qk_and_offsets()
                last = new_bs - 1
                L = int(self.tree_value_prefix_lens[last])
                N = int(self.tree_value_candidate_lens[last])
                P = int(self.tree_value_cached_prefix_lens[last])
                q = (L - P) + N
                k = L + N
                end_mask = int(mask_offsets[last]) + int(q * k)
                end_pos = int(pos_offsets[last]) + int(q)

                self.custom_mask = self.custom_mask.narrow(0, 0, end_mask)
                self.positions = self.positions.narrow(0, 0, end_pos)
                self.tree_value_prefix_lens = self.tree_value_prefix_lens[:new_bs]
                self.tree_value_candidate_lens = self.tree_value_candidate_lens[:new_bs]
                self.tree_value_cached_prefix_lens = self.tree_value_cached_prefix_lens[:new_bs]
                return
            # Fall through to the general gather-based path below.

        q_lens, _k_lens, mask_offsets, pos_offsets = self._per_req_qk_and_offsets()

        mask_chunks: List[torch.Tensor] = []
        pos_chunks: List[torch.Tensor] = []
        new_Ls: List[int] = []
        new_Ns: List[int] = []
        new_Ps: List[int] = []

        for i in idx_list:
            i = int(i)
            q = int(q_lens[i])
            m0 = int(mask_offsets[i])
            p0 = int(pos_offsets[i])
            # mask len = q*k, but we only need the flat length; compute it from offsets:
            L = int(self.tree_value_prefix_lens[i])
            N = int(self.tree_value_candidate_lens[i])
            k = L + N
            ml = q * k
            mask_chunks.append(self.custom_mask.narrow(0, m0, ml))
            pos_chunks.append(self.positions.narrow(0, p0, q))
            new_Ls.append(L)
            new_Ns.append(N)
            new_Ps.append(int(self.tree_value_cached_prefix_lens[i]))

        self.custom_mask = torch.cat(mask_chunks, dim=0) if mask_chunks else self.custom_mask[:0]
        self.positions = torch.cat(pos_chunks, dim=0) if pos_chunks else self.positions[:0]
        self.tree_value_prefix_lens = new_Ls
        self.tree_value_candidate_lens = new_Ns
        self.tree_value_cached_prefix_lens = new_Ps

    def merge_batch(self, spec_info: "TreeValueSpecInput"):
        """
        Merge another TreeValueSpecInput into this one (used by ScheduleBatch.merge_batch()).
        """
        if spec_info is None:
            return
        self.custom_mask = torch.cat([self.custom_mask, spec_info.custom_mask], dim=0)
        self.positions = torch.cat([self.positions, spec_info.positions], dim=0)
        self.tree_value_prefix_lens.extend(list(spec_info.tree_value_prefix_lens))
        self.tree_value_candidate_lens.extend(list(spec_info.tree_value_candidate_lens))
        self.tree_value_cached_prefix_lens.extend(list(spec_info.tree_value_cached_prefix_lens))


def build_tree_value_custom_mask_and_positions(
    *,
    prefix_lens: List[int],
    candidate_lens: List[int],
    cached_prefix_lens: List[int],
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Build flattened custom_mask and positions for a batch.

    For each request:
      - cached prefix length: P (len(req.prefix_indices))
      - full prefix length: L
      - candidates count: N (1-token each)
      - extend tokens length: Q = (L - P) + N
      - total K length: K = (P + Q) = L + N

    Tree attention:
      - Prefix (the new prefix tokens in extend) are causal within (cached+new) prefix.
      - Each candidate attends to all prefix tokens (cached+new) and itself only.
    Positions:
      - cached+new prefix positions: 0..L-1 (cached part isn't forwarded, but positions for new tokens continue)
      - candidate positions: L (shared)

    Note: returned `positions` corresponds to extend tokens only (length sum(Q)).
    """

    assert len(prefix_lens) == len(candidate_lens) == len(cached_prefix_lens)
    bs = len(prefix_lens)

    import numpy as np

    if bs == 0:
        return (
            torch.empty((0,), dtype=torch.bool, device=device),
            torch.empty((0,), dtype=torch.int64, device=device),
        )

    Ls = [int(x) for x in prefix_lens]
    Ns = [int(x) for x in candidate_lens]
    Ps = [int(x) for x in cached_prefix_lens]

    for i, (L, N, P) in enumerate(zip(Ls, Ns, Ps)):
        if L <= 0:
            raise ValueError("prefix length must be > 0")
        if N <= 0:
            raise ValueError("candidate length must be > 0")
        if P < 0 or P > L:
            raise ValueError(f"cached prefix len {P} must be in [0, {L}]")

    q_lens = [(L - P) + N for L, N, P in zip(Ls, Ns, Ps)]
    k_lens = [L + N for L, N in zip(Ls, Ns)]
    total_mask = sum(q * k for q, k in zip(q_lens, k_lens))
    total_pos  = sum(q_lens)

    # Build entirely on CPU; single GPU transfer at the end.
    # Replaces O(bs * N) small GPU kernel launches with one PCIe transfer.
    mask_buf = np.zeros(total_mask, dtype=np.bool_)
    pos_buf  = np.empty(total_pos,  dtype=np.int64)

    mask_off = 0
    pos_off  = 0
    for L, N, P, q_len, k_len in zip(Ls, Ns, Ps, q_lens, k_lens):
        # Row-major [q_len, k_len] view into mask_buf.
        m = mask_buf[mask_off : mask_off + q_len * k_len].reshape(q_len, k_len)

        # Prefix tail rows: causal attendance within cached+new prefix.
        for t in range(L - P):
            abs_q = P + t
            m[t, : abs_q + 1] = True

        # Candidate rows: attend all prefix tokens (0..L-1) and itself.
        cand_row_start = L - P
        m[cand_row_start : cand_row_start + N, :L] = True
        for j in range(N):
            m[cand_row_start + j, L + j] = True

        mask_off += q_len * k_len

        # Position entries: [P, P+1, ..., L-1] + [L]*N
        new_prefix_len = L - P
        if new_prefix_len > 0:
            pos_buf[pos_off : pos_off + new_prefix_len] = np.arange(P, L, dtype=np.int64)
        pos_buf[pos_off + new_prefix_len : pos_off + q_len] = L
        pos_off += q_len

    custom_mask = torch.from_numpy(mask_buf).to(device=device, non_blocking=True)
    positions   = torch.from_numpy(pos_buf).to(device=device, non_blocking=True)
    return custom_mask, positions

