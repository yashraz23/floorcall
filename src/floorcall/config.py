"""Every threshold and knob in floorcall. No magic numbers anywhere else.

Values are experiment variables. Change them here, never inline, and say so in the commit body
(CLAUDE.md §15). Any value can be overridden from the environment with the `FLOORCALL_` prefix and
`__` for nesting, e.g. `FLOORCALL_POLICY__THETA_INTERRUPT=0.7`.
"""

from functools import lru_cache
from pathlib import Path

from pydantic import BaseModel, Field, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

REPO_ROOT = Path(__file__).resolve().parents[2]


class LayaSettings(BaseModel):
    """Which checkpoint answers the questions, pinned to an exact Hub commit."""

    checkpoint: str = "convaiinnovations/laya"
    # The English checkpoint's Hub commit on 2026-09-27. Pinned so that a re-upload upstream cannot
    # move a reported number. Its rl_agent_config.json gives max_len=512, head_max_len=192.
    revision: str | None = "55cf4c4ebb4ebe31b2550e8bdf3bd21b99753851"
    # None lets Laya pick (CUDA if it can place the model there). The latency harness pins it.
    device: str | None = None


class StateSettings(BaseModel):
    """The state packer (floorcall.state)."""

    # Tokens held back below the smallest room any question in an event leaves for the state.
    # The room is measured exactly by the adapter; the margin only absorbs tokenizer merges across
    # a boundary when a field is shortened, which can shift a count by a token or two.
    safety_margin_tokens: int = 8
    # ASR-style normalization of every text field. False only for the Table D ablation that shows
    # what punctuation leakage does; never for a reported main-table number.
    normalize: bool = True


class PolicySettings(BaseModel):
    """Thresholds that turn probabilities into actions (floorcall.policy).

    Provisional. Each is a point on a tradeoff curve and is chosen on the calibration split in
    milestone 2, never on a test split. Until then they sit at 0.5, which makes no claim.
    """

    theta_yield: float = Field(0.5, ge=0.0, le=1.0)
    # The safety net: respond after this much silence whatever p(turn_complete) says.
    max_wait_ms: int = Field(2000, gt=0)
    theta_escalate: float = Field(0.5, ge=0.0, le=1.0)
    theta_oos: float = Field(0.5, ge=0.0, le=1.0)
    theta_interrupt: float = Field(0.5, ge=0.0, le=1.0)


class SplitSettings(BaseModel):
    """Conversation-disjoint splitting (floorcall.data.splits)."""

    seed: int = 20260927
    train: float = 0.70
    # Held out for temperature fitting and threshold choice. Never trained on, never a test set.
    calib: float = 0.10
    test: float = 0.20

    @model_validator(mode="after")
    def _fractions_sum_to_one(self) -> "SplitSettings":
        total = self.train + self.calib + self.test
        if abs(total - 1.0) > 1e-9:
            raise ValueError(f"split fractions must sum to 1, got {total}")
        return self


class DataSettings(BaseModel):
    """Dataset construction (floorcall.data). Changing any of these changes the frozen test sets."""

    # Merged turns of history kept per example before the packer trims to the token budget.
    max_history_turns: int = 8
    # SwDA tags short listener responses by function: aa (agree), bk (acknowledge), ba
    # (appreciate), na/ny (yes-answers). 81-100% of them are three words or fewer ("yeah",
    # "right", "okay", "uh huh"), and at that length they do not take the floor any more than a
    # "b" backchannel does. Up to this many words, they do not count as taking or holding it.
    minimal_response_max_words: int = 3
    # D2: a barge-in is decided on the first words the recogniser has, not the whole turn. 12
    # words is about 4 s of speech, enough to hold "yeah but that's not what i asked".
    d2_partial_words: int = 12
    # D2 hard subset: a backchannel form qualifies when at least this many train backchannels
    # are exactly it AND at least this many train interruptions open with it ...
    d2_hard_min_count: int = 20
    # ... and neither class dominates it: each is at least this share of the form's train rows.
    # Without it, "uh huh" qualifies on volume alone (8% of its rows are interruptions) and the
    # "hard" subset becomes most of the test set.
    d2_hard_min_share: float = 0.2
    # D1 hard subset: last words seen at least this often ending train truncations are rated.
    d1_hard_min_count: int = 10


class EvalSettings(BaseModel):
    """Metrics and the latency benchmark."""

    # 15 equal-width bins, the convention laya.common.ece_score uses, so ECE here is comparable
    # to the figures Laya publishes.
    ece_bins: int = 15
    latency_warmup: int = 50
    latency_iters: int = 1000
    gpu_p99_budget_ms: float = 50.0
    cpu_p99_budget_ms: float = 100.0


class TrainSettings(BaseModel):
    """RLCD fine-tuning (floorcall.train), reproducing Laya's Kaggle notebook recipe.

    The notebook ran 2 GPUs x micro-batch 8 x accumulation 4 = 64 rows per update. One GPU
    reaches the same 64 with accumulation 8, so the learning-rate schedule sees the same batch.
    """

    epochs: int = 4
    micro_batch: int = 8
    grad_accum: int = 8
    group_size: int = 4  # GRPO baseline samples per row
    lr_encoder: float = 2.5e-5
    lr_head: float = 1.0e-4
    lr_min: float = 1.0e-6
    weight_decay: float = 0.01
    sigma_start: float = 0.4  # exploration noise on the logits, annealed linearly per epoch
    sigma_end: float = 0.1
    w_sph: float = 0.75  # spherical-score weight in the reward
    w_rps: float = 1.0  # ranked-probability weight (score questions only)
    ce_weight: float = 1.0  # soft cross-entropy guidance next to the policy-gradient term
    clip_grad_norm: float = 1.0
    # bf16 on Blackwell (CLAUDE.md §11). The notebook used fp16 + GradScaler because T4s have no
    # bf16; bf16's range makes the scaler unnecessary.
    amp_dtype: str = "bf16"
    seed: int = 20260927


class PathSettings(BaseModel):
    data_raw: Path = REPO_ROOT / "data" / "raw"
    data_processed: Path = REPO_ROOT / "data" / "processed"
    test_frozen: Path = REPO_ROOT / "data" / "test_frozen"


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="FLOORCALL_",
        env_nested_delimiter="__",
        env_file=REPO_ROOT / ".env",
        extra="ignore",
    )

    laya: LayaSettings = LayaSettings()
    state: StateSettings = StateSettings()
    policy: PolicySettings = PolicySettings()
    splits: SplitSettings = SplitSettings()
    data: DataSettings = DataSettings()
    eval: EvalSettings = EvalSettings()
    train: TrainSettings = TrainSettings()
    paths: PathSettings = PathSettings()


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
