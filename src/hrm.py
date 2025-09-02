from dataclasses import dataclass
from typing import Optional, Tuple

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.layers import RMSNorm, TransformerEncoder


# -------------------------
# HRM core
# -------------------------


@dataclass
class HRMConfig:
    vocab_size: int
    d_model: int = 512
    n_heads: int = 8
    depth_L: int = 4
    depth_H: int = 4
    max_len: int = 1024
    N_cycles: int = 2  # high-level cycles
    T_steps: int = 2  # low-level steps per cycle
    stablemax: bool = False  # use softmax by default
    act_enabled: bool = True  # simple ACT-style halting head
    q_epsilon: float = 0.1  # exploration prob for ACT
    q_hidden: int = 128  # Q head size


class HRM(nn.Module):
    def __init__(self, cfg: HRMConfig):
        super().__init__()
        self.cfg = cfg
        self.tok_emb = nn.Embedding(cfg.vocab_size, cfg.d_model)

        # Recurrent modules
        self.L_net = TransformerEncoder(
            cfg.d_model, cfg.n_heads, cfg.depth_L, rope=True
        )
        self.H_net = TransformerEncoder(
            cfg.d_model, cfg.n_heads, cfg.depth_H, rope=True
        )

        # Output head over tokens (per-position). Use first (CLS) by default.
        self.out_norm = RMSNorm(cfg.d_model)
        self.out_proj = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)

        # Optional Q-head for simple ACT
        if cfg.act_enabled:
            self.q_head = nn.Sequential(
                RMSNorm(cfg.d_model),
                nn.Linear(cfg.d_model, cfg.q_hidden),
                nn.GELU(),
                nn.Linear(cfg.q_hidden, 2),  # [halt, continue]
            )
        else:
            self.q_head = None

        self.z0H = nn.Parameter(torch.empty(1, 1, cfg.d_model))
        self.z0L = nn.Parameter(torch.empty(1, 1, cfg.d_model))
        nn.init.trunc_normal_(self.z0H, std=1.0 /
                              math.sqrt(cfg.d_model), a=-2.0, b=2.0)
        nn.init.trunc_normal_(self.z0L, std=1.0 /
                              math.sqrt(cfg.d_model), a=-2.0, b=2.0)
        self.z0H.requires_grad_(False)
        self.z0L.requires_grad_(False)

        self._init_params()

    def _init_params(self):
        # Truncated Lecun normal-esque init for linear/embeds
        for m in self.modules():
            if isinstance(m, (nn.Linear, nn.Embedding)):
                nn.init.trunc_normal_(
                    m.weight, std=1.0 / math.sqrt(m.weight.shape[-1]), a=-2.0, b=2.0
                )
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.zeros_(m.bias)

    # ------------
    # Helper heads
    # ------------
    def decode_token_logits(self, zH_seq: torch.Tensor) -> torch.Tensor:
        """Decode per-position logits for language modeling: (B, L, vocab)."""
        h = self.out_norm(zH_seq)
        return self.out_proj(h)

    def q_values(self, zH_seq: torch.Tensor) -> torch.Tensor:
        # Predict [halt, continue] from CLS/pooled high-level state
        if not self.cfg.act_enabled:
            raise RuntimeError("Q-head called but ACT is disabled")

        h = zH_seq.mean(dim=1)
        return self.q_head(h)

    # --------------------
    # One forward "segment"
    # --------------------
    def hrm_segment(
        self, zH: torch.Tensor, zL: torch.Tensor, x_emb: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        """
        Run one HRM segment using the "one-step gradient" approximation.
        Returns (zH_new, zL_new, logits, q_values_or_None).

        Mechanism:
        - Run N*T - 1 steps without grad to approach local equilibria
        - Run 1 final step with grad enabled
        """
        N, T = self.cfg.N_cycles, self.cfg.T_steps
        total = N * T

        # --- no-grad steps ---
        with torch.no_grad():
            zH_ng = zH
            zL_ng = zL
            for i in range(total - 1):
                # low-level always updates
                zL_ng = self.L_net(zL_ng + zH_ng + x_emb)
                # high-level updates every T steps
                if ((i + 1) % T) == 0:
                    zH_ng = self.H_net(zH_ng + zL_ng)

        # --- final step with grad ---
        zL = self.L_net(zL_ng + zH_ng + x_emb)
        zH = self.H_net(zH_ng + zL)

        # For LM, we prefer per-token logits
        logits = self.decode_token_logits(zH)
        qv = self.q_values(zH) if self.cfg.act_enabled else None
        return zH, zL, logits, qv

    # --------------------
    # Public forward API
    # --------------------
    def forward(
        self,
        x: torch.Tensor,
        segments: int = 1,
        teacher: Optional[torch.Tensor] = None,
        act_infer_max: Optional[int] = None,
    ):
        """
        x: (B, L) integer tokens
        segments: number of deep supervision segments M
        teacher: optional LM targets (B, L) integer class ids
        act_infer_max: if ACT enabled, limit for inference-time scaling (Mmax). If None, use segments.
        """
        B, L = x.shape
        x_emb = self.tok_emb(x)  # (B, L, D)

        zH = self.z0H.expand(B, L, -1).to(x.device)
        zL = self.z0L.expand(B, L, -1).to(x.device)

        seg_outputs = []
        seg_qvals = []
        ce_sum = None  # accumulate CE over segments

        # use 'segments' for training; only use act_infer_max for inference
        M = segments if (
            teacher is not None or act_infer_max is None) else act_infer_max

        m = 0
        while m < M:
            zH, zL, logits, qv = self.hrm_segment(zH, zL, x_emb)
            seg_outputs.append(logits)           # (B, L, V)
            if qv is not None:
                seg_qvals.append(qv)             # (B, 2)

            # Deep supervision CE
            if teacher is not None:
                # If you have PADs, do: ignore_index=self.cfg.pad_token_id
                ce = F.cross_entropy(logits.transpose(1, 2), teacher)
                ce_sum = ce if (ce_sum is None) else (ce_sum + ce)

            # 1-step gradient approx across segments
            zH = zH.detach()
            zL = zL.detach()

            # ACT early halt *only in inference*
            if self.cfg.act_enabled and (teacher is None) and (act_infer_max is not None):
                with torch.no_grad():
                    probs = torch.softmax(qv, dim=-1)  # (B,2)
                    halt_majority = (
                        probs[:, 0] > probs[:, 1]).float().mean() > 0.5
                if halt_majority:
                    break
            m += 1

        total_segments = len(seg_outputs)

        # ----- ACT Q-learning loss (per-segment), average ONCE over segments -----
        q_loss_sum = None
        if self.cfg.act_enabled and (teacher is not None):
            # (Optional) PAD-aware exact-match for the halt reward
            pad_id = getattr(self.cfg, "pad_token_id", None)
            for i in range(total_segments):
                qv_i = seg_qvals[i]              # (B,2)
                logits_i = seg_outputs[i]        # (B,L,V)

                preds_i = logits_i.argmax(dim=-1)  # (B,L)
                if pad_id is not None:
                    mask = (teacher != pad_id)
                    # exact match on non-PAD positions
                    correct_all = ((preds_i == teacher) | (~mask)).all(dim=-1)
                else:
                    correct_all = (preds_i == teacher).all(dim=-1)
                reward_halt = correct_all.float()  # (B,)

                # continue target: bootstrap from next step; at last step, "forced halt"
                if i < total_segments - 1:
                    with torch.no_grad():
                        cont_tgt = torch.sigmoid(
                            seg_qvals[i + 1]).max(dim=-1).values  # (B,)
                else:
                    with torch.no_grad():
                        # use halt prob at M_max
                        cont_tgt = torch.sigmoid(qv_i[:, 0])

                target = torch.stack(
                    [reward_halt, cont_tgt], dim=-1)  # (B,2) in [0,1]
                bce_i = F.binary_cross_entropy_with_logits(
                    qv_i, target)  # logits vs probs

                q_loss_sum = bce_i if (
                    q_loss_sum is None) else (q_loss_sum + bce_i)

        # ---- combine and average ONCE over segments ----
        total_loss = None
        if (ce_sum is not None) and (q_loss_sum is not None):
            total_loss = (ce_sum + q_loss_sum) / total_segments
        elif ce_sum is not None:
            total_loss = ce_sum / total_segments
        elif q_loss_sum is not None:
            total_loss = q_loss_sum / total_segments

        out = {
            "logits": seg_outputs[-1],  # (B, L, V)
            "all_segment_logits": seg_outputs,
            "loss": total_loss,
        }
        if self.cfg.act_enabled:
            out["qvals"] = seg_qvals
        return out
