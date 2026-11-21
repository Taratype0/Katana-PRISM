"""Self-contained width-batched PWM scanner for scsplicetrace."""
from __future__ import annotations

import numpy as np

_ENCODE_TABLE = np.full(256, 4, dtype=np.int16)
_ENCODE_TABLE[np.frombuffer(b"ACGT", dtype=np.uint8)] = np.arange(4)


def batch_pwm_profiles_for_width(
    sequences: dict[str, str],
    odds_batch: np.ndarray,
    w: int,
    widths: tuple[int, ...] = (0,),
) -> dict[str, dict[int, np.ndarray]]:
    m_count = len(odds_batch)
    result: dict[str, dict[int, np.ndarray]] = {}
    for ev_id, raw_seq in sequences.items():
        seq = raw_seq.upper().replace("U", "T")
        seq_len = len(seq)
        n = max(0, seq_len - w + 1)
        ev_out: dict[int, np.ndarray] = {}
        if n == 0:
            for width in widths:
                span = 1 if width == 0 else max(0, seq_len - width + 1)
                ev_out[width] = np.full((m_count, span), np.nan, dtype=float)
            result[ev_id] = ev_out
            continue
        encoded = _ENCODE_TABLE[np.frombuffer(seq.encode("ascii"), dtype=np.uint8)]
        invalid = np.r_[0, np.cumsum(encoded == 4)]
        valid = (invalid[w:] - invalid[:-w]) == 0
        starts = np.flatnonzero(valid)
        if len(starts):
            vals = np.ones((m_count, len(starts)), dtype=float)
            for k in range(w):
                vals *= odds_batch[:, k, encoded[starts + k]]
            mean_log = np.log2(1.0 + np.mean(vals, axis=1))[:, None]
            for width in widths:
                if width == 0:
                    ev_out[0] = mean_log
                else:
                    span = max(0, seq_len - width + 1)
                    if span == 0 or width < w:
                        ev_out[width] = np.full((m_count, span), np.nan, dtype=float)
                    else:
                        full_grid = np.zeros((m_count, n), dtype=float)
                        full_grid[:, starts] = vals
                        v_grid = np.zeros(n, dtype=float)
                        v_grid[starts] = 1.0
                        win_k = width - w + 1
                        cs_val = np.concatenate([np.zeros((m_count, 1)), np.cumsum(full_grid, axis=1)], axis=1)
                        cs_cnt = np.concatenate([[0.0], np.cumsum(v_grid)])
                        w_sum = cs_val[:, win_k:] - cs_val[:, :-win_k]
                        w_cnt = cs_cnt[win_k:] - cs_cnt[:-win_k]
                        w_mean = np.divide(w_sum, w_cnt[None, :], out=np.full((m_count, span), np.nan), where=w_cnt[None, :] > 0)
                        ev_out[width] = np.where(np.isfinite(w_mean), np.log2(1.0 + w_mean), np.nan)
        else:
            for width in widths:
                span = 1 if width == 0 else max(0, seq_len - width + 1)
                ev_out[width] = np.full((m_count, span), np.nan, dtype=float)
        result[ev_id] = ev_out
    return result
