"""
scorer.py — continuous stance scoring for Type-L agent opinions.

Type-L agents emit free text; their position on [-1, +1] comes from scoring that
text, not from asking the model for a number. This removes the 5-level
quantisation that previously drove tie-heavy K-NN rewiring, and avoids the
central-tendency and round-number biases documented for direct numeric
elicitation from LLMs.

The scorer is pluggable so the simulation does not change when the Stage-1
regressor is retrained:

    RobertaScorer     the Stage-1 fine-tuned stance regressor — the intended
                      configuration. Same instrument as was applied to the
                      Reddit corpus, so agent positions and empirical positions
                      live on one scale.
    SelfReportScorer  transitional fallback that parses a number out of the
                      text. Documented to be biased; valid only until the
                      Stage-1 weights exist, and never as the headline
                      configuration.

Scoring is batched: the model collects every agent's opinion for a step and
scores them in one forward pass.
"""

from __future__ import annotations

import re
from typing import Sequence


class StanceScorer:
    """Maps opinion texts to continuous positions in [-1, +1]."""

    name = "base"

    def score(self, texts: Sequence[str]) -> list[float]:
        raise NotImplementedError

    @staticmethod
    def _clip(x: float) -> float:
        return max(-1.0, min(1.0, float(x)))


class RobertaScorer(StanceScorer):
    """Stage-1 fine-tuned stance regressor (`Reddit-Dataset/model/final_model.pt`).

    Reproduces the training-time architecture exactly: RoBERTa encoder, CLS
    token, a single linear head, and tanh — so the checkpoint loads without
    remapping and inference matches training.
    """

    name = "roberta"

    def __init__(self, ckpt_path: str, device: str = "cuda", batch_size: int = 64,
                 max_length: int = 256):
        import torch
        import torch.nn as nn
        from transformers import RobertaModel, RobertaTokenizer

        self._torch = torch
        self.batch_size = batch_size
        self.max_length = max_length
        self.device = device

        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        roberta_name = ckpt.get("roberta_name", "roberta-base")
        self.cv_mean_mse = ckpt.get("cv_mean_mse")

        class _StanceModel(nn.Module):
            def __init__(self, name):
                super().__init__()
                self.roberta = RobertaModel.from_pretrained(name)
                self.regressor = nn.Linear(768, 1)

            def forward(self, input_ids, attention_mask):
                out = self.roberta(input_ids=input_ids, attention_mask=attention_mask)
                cls = out.last_hidden_state[:, 0, :]
                return torch.tanh(self.regressor(cls)).squeeze(-1)

        self.tokenizer = RobertaTokenizer.from_pretrained(roberta_name)
        self.model = _StanceModel(roberta_name)
        self.model.load_state_dict(ckpt["model_state"])
        self.model.to(device).eval()

    def score(self, texts: Sequence[str]) -> list[float]:
        torch = self._torch
        out: list[float] = []
        with torch.no_grad():
            for i in range(0, len(texts), self.batch_size):
                batch = [t or "" for t in texts[i:i + self.batch_size]]
                enc = self.tokenizer(batch, return_tensors="pt", padding=True,
                                     truncation=True, max_length=self.max_length)
                enc = {k: v.to(self.device) for k, v in enc.items()}
                preds = self.model(**enc)
                if preds.dim() == 0:
                    preds = preds.unsqueeze(0)
                out.extend(self._clip(v) for v in preds.tolist())
        return out


class SelfReportScorer(StanceScorer):
    """Transitional fallback: reads a number out of the text.

    Direct numeric elicitation from an LLM regresses toward the centre of the
    scale and clusters on round values. Use only while the Stage-1 weights are
    unavailable, and record it in the run metadata so such results are not
    silently compared against RoBERTa-scored runs.
    """

    name = "self_report"
    _NUM = re.compile(r"[-+]?\d*\.?\d+")

    def score(self, texts: Sequence[str]) -> list[float]:
        out = []
        for t in texts:
            m = self._NUM.search(t or "")
            out.append(self._clip(float(m.group())) if m else 0.0)
        return out


def build_scorer(kind: str, **kwargs) -> StanceScorer:
    if kind == "roberta":
        return RobertaScorer(**kwargs)
    if kind == "self_report":
        return SelfReportScorer()
    raise ValueError(f"unknown scorer: {kind!r}")
