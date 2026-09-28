"""Every threshold and knob in floorcall. No magic numbers anywhere else.

Values are experiment variables. Change them here, never inline, and say so in the commit body
(CLAUDE.md §15). Any value can be overridden from the environment with the `FLOORCALL_` prefix and
`__` for nesting, e.g. `FLOORCALL_POLICY__THETA_INTERRUPT=0.7`.
"""

from functools import lru_cache
from pathlib import Path

from pydantic import BaseModel, Field, SecretStr, model_validator
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
    # Replay the forward as CUDA graphs (DECISIONS.md D-012). Sequence lengths are padded up to a
    # multiple of this many tokens, so one graph serves every state in its bucket.
    cuda_graphs: bool = False
    graph_bucket_tokens: int = 32


class StateSettings(BaseModel):
    """The state packer (floorcall.state)."""

    # Tokens held back below the smallest room any question in an event leaves for the state.
    # The room is measured exactly by the adapter; the margin only absorbs tokenizer merges across
    # a boundary when a field is shortened, which can shift a count by a token or two.
    safety_margin_tokens: int = 8
    # ASR-style normalization of every text field. False only for the Table D ablation that shows
    # what punctuation leakage does; never for a reported main-table number.
    normalize: bool = True
    # Table D ablations: leave the conversation history, or the agent's last utterance, out of
    # every state, in training and evaluation alike. True for every main-table number.
    include_history: bool = True
    include_agent: bool = True


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
    # Silence after which VAD raises user_pause. It sets the floor of the response delay, and
    # the delay model of the theta_yield curve (floorcall.evaluate.curves).
    vad_pause_ms: int = Field(300, gt=0)
    # How theta_interrupt and theta_yield are chosen on calib: the smallest theta whose rate of
    # the costly error stays at or under these (floorcall.evaluate.curves.choose_threshold).
    target_false_stop_rate: float = Field(0.05, ge=0.0, le=1.0)
    target_premature_rate: float = Field(0.05, ge=0.0, le=1.0)


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
    # D4 hand labelling. Test and calib candidates are drawn from different threads; the test set
    # needs at least d4_min_test_labels true/false labels (CLAUDE.md §4). The margin over it
    # absorbs skips. Calib labels are what theta_escalate and the D4 temperature are fitted on,
    # since neither may be chosen on test.
    d4_candidates_test: int = 400
    d4_candidates_calib: int = 200
    d4_min_test_labels: int = 300


class EvalSettings(BaseModel):
    """Metrics and the latency benchmark."""

    # 15 equal-width bins, the convention laya.common.ece_score uses, so ECE here is comparable
    # to the figures Laya publishes.
    ece_bins: int = 15
    # Bounds on a fitted temperature (floorcall.evaluate.calibration). A fit that lands on one is
    # flagged: on data the model separates perfectly, the unbounded optimum is T -> 0.
    temperature_min: float = 0.25
    temperature_max: float = 10.0
    # Reliability figures leave off bins with fewer rows than this: a two-row bin swings between
    # accuracy 0 and 1 and says nothing. ECE itself always uses every row.
    reliability_min_bin_count: int = 20
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
    # Rows drawn per task per epoch (floorcall.train.data.epoch_mixture). The raw train sets differ
    # 30x in size; these quotas keep the checkpoint multi-task. D3 (1,750 rows) is repeated 3x per
    # epoch. The escalate quota is provisional until D4's training source is decided (D-009).
    rows_per_epoch: dict[str, int] = Field(
        default_factory=lambda: {
            "turn_complete": 20_000,
            "barge_in": 20_000,
            "route": 5_250,
            "escalate": 5_000,
        }
    )
    # Gold labels here are hard; 0 keeps the target one-hot. Temperature scaling on calib, not
    # smoothing, is what calibrates the output.
    label_smoothing: float = 0.0
    log_every_updates: int = 10


class ProviderPin(BaseModel):
    """The one OpenRouter endpoint a model is served from, with no fallback (D-032)."""

    # OpenRouter's endpoint slug, provider plus variant. A bare provider slug would match every
    # endpoint that provider runs, whatever its quantization.
    endpoint: str
    # The provider name OpenRouter reports in a response's "provider" field; a response naming any
    # other provider is rejected.
    name: str
    quantization: str


class LLMSettings(BaseModel):
    """LLMs via OpenRouter: the D4 training labeller and the prompted-LLM baseline (D-030, D-031)."""

    base_url: str = "https://openrouter.ai/api/v1"
    # Exact model ids, never the ":batch" variants OpenRouter also lists.
    labeller_model: str = "openai/gpt-oss-120b"
    baseline_model: str = "openai/gpt-oss-20b"
    reasoning_effort: str = "low"
    seed: int = 20260927
    # Ceiling on what OpenRouter may route to, USD per million tokens (prompt, completion), sent as
    # `provider.max_price` so no provider above it is used. These are the rates the budget was
    # planned at. They are also each call's worst case when it is reserved against the budget; what
    # a call is charged is OpenRouter's own `usage.cost`. A model missing here is refused, because
    # its spend could not be bounded.
    max_price_per_million: dict[str, tuple[float, float]] = Field(
        default_factory=lambda: {
            "openai/gpt-oss-120b": (0.15, 0.60),
            "openai/gpt-oss-20b": (0.075, 0.30),
        }
    )
    # Models served from one fixed endpoint (D-032). The labeller runs on Crusoe's bf16 build of
    # gpt-oss-120b only: no other provider, no other quantization. The baseline is not pinned.
    provider_pins: dict[str, ProviderPin] = Field(
        default_factory=lambda: {
            "openai/gpt-oss-120b": ProviderPin(
                endpoint="crusoe/bf16", name="Crusoe", quantization="bf16"
            )
        }
    )
    # A labeller prompt is accepted only if, on calib, it agrees with Yash's hand labels at least
    # this well (D-032). Test and train are labelled only with an accepted prompt.
    labeller_min_kappa: float = 0.60
    labeller_min_escalate_precision: float = 0.70
    # Yash's cap for all LLM use in this project: $5, with the stop at 95% of it, $4.75. A call is
    # refused unless its worst case still fits under the stop, counting everything already spent.
    budget_usd: float = 5.0
    budget_margin: float = 0.95
    # Reasoning tokens are billed and count against this even when excluded from the response.
    max_output_tokens: int = 800
    concurrency: int = 4
    max_retries: int = 6
    timeout_s: float = 60.0
    # D4 training rows to label from train-split banking threads, one message per thread.
    d4_train_rows: int = 6000
    # Prompted-LLM baseline: D1 and D2 test sets are subsampled (evenly spaced, fixed); D3 and D4
    # are scored whole. The n is reported beside every number.
    baseline_rows: dict[str, int] = Field(
        default_factory=lambda: {"turn_complete": 1000, "barge_in": 1000}
    )


class PathSettings(BaseModel):
    data_raw: Path = REPO_ROOT / "data" / "raw"
    data_processed: Path = REPO_ROOT / "data" / "processed"
    test_frozen: Path = REPO_ROOT / "data" / "test_frozen"
    labels: Path = REPO_ROOT / "data" / "labels"


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
    llm: LLMSettings = LLMSettings()
    paths: PathSettings = PathSettings()
    # Read from the environment or .env, never logged or written anywhere. SecretStr keeps it out
    # of settings dumps (run.json stores one).
    openrouter_api_key: SecretStr | None = Field(
        default=None, validation_alias="OPENROUTER_API_KEY"
    )


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()


# Table D (CLAUDE.md §12): each ablation is a named set of StateSettings overrides, applied to
# training, calibration and evaluation alike and recorded in the run's run.json.
ABLATIONS: dict[str, dict[str, bool]] = {
    "full": {},
    "no_history": {"include_history": False},
    "no_agent": {"include_agent": False},
    "no_normalize": {"normalize": False},
}


def with_ablation(settings: Settings, name: str) -> Settings:
    if name not in ABLATIONS:
        raise ValueError(f"unknown ablation {name!r}; one of {sorted(ABLATIONS)}")
    state = settings.state.model_copy(update=ABLATIONS[name])
    return settings.model_copy(update={"state": state})
