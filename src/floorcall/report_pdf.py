"""The project report as a PDF (DECISIONS.md D-044, D-051).

    uv run floorcall release report        # docs/report/floorcall_report.pdf, from a clean commit

Every number in the report is read from a committed file when the report is built: the results
files, the dataset cards, the label files, the replay run. None is typed into this module. A
number whose file is missing prints as TODO. The text around the numbers explains them.

`content()` builds the report as a list of blocks from those files, and is tested without drawing
anything. `render()` draws the blocks with fpdf2, using the DejaVu fonts that matplotlib ships, so
the PDF looks the same on any machine. The PDF's creation date is the commit's date, so building
twice from one commit gives the same bytes.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

from floorcall.config import REPO_ROOT, LayaSettings

RESULTS = REPO_ROOT / "results"
REPORT = REPO_ROOT / "docs" / "report" / "floorcall_report.pdf"
TODO = "TODO"
GITHUB = "https://github.com/yashraz23/floorcall"
MODEL = "https://huggingface.co/enz23/floorcall"
SPACE = "https://huggingface.co/spaces/enz23/floorcall"
DECISIONS = ("turn_complete", "barge_in", "route", "escalate")
SOURCE_NAMES = {"swda": "SwDA", "clinc": "CLINC150", "twcs": "Twitter support"}
BASELINE_NAMES = {"tfidf_lr": "TF-IDF + logistic regression", "lexical_rule": "a lexical rule"}
NAMES = {"turn_complete": "D1 turn_complete", "barge_in": "D2 barge_in", "route": "D3 route",
         "escalate": "D4 escalate"}  # fmt: skip


@dataclass
class Block:
    kind: str  # h1 | h2 | p | bullets | table | image | code | meta
    text: str = ""
    items: list[str] = field(default_factory=list)
    header: list[str] = field(default_factory=list)
    rows: list[list[str]] = field(default_factory=list)
    widths: list[float] = field(default_factory=list)
    path: Path | None = None
    size: float = 7.5


# -- reading committed files ------------------------------------------------------------------


def _load(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None


def _num(path: Path, *keys: str | int, fmt: str = "{:.3f}") -> str:
    """A number from a JSON file, formatted, or TODO if the file or a key is missing."""
    v: Any = _load(path)
    for k in keys:
        in_dict = isinstance(v, dict) and k in v
        in_list = isinstance(v, list) and isinstance(k, int) and -len(v) <= k < len(v)
        if not (in_dict or in_list):
            return TODO
        v = v[k]
    return fmt.format(v) if isinstance(v, int | float) else TODO


def _ci(path: Path, metric: str) -> str:
    """'0.937 [0.931, 0.943]': a Table A metric with its bootstrap interval (`metrics.ci95`)."""
    point = _num(path, "metrics", metric)
    lo, hi = _num(path, "metrics", "ci95", metric, 0), _num(path, "metrics", "ci95", metric, 1)
    return point if TODO in (lo, hi) else f"{point} [{lo}, {hi}]"


def _diff(path: Path, metric: str) -> str:
    """'+0.029 [+0.022, +0.035]' from a paired-comparison file."""
    point = _num(path, "differences", metric, "difference", fmt="{:+.3f}")
    lo = _num(path, "differences", metric, "ci95", 0, fmt="{:+.3f}")
    hi = _num(path, "differences", metric, "ci95", 1, fmt="{:+.3f}")
    return TODO if TODO in (point, lo, hi) else f"{point} [{lo}, {hi}]"


def md_table(md: str) -> tuple[list[str], list[list[str]]]:
    """A pipe table from the README renderer as (header, rows)."""
    lines = [ln for ln in md.splitlines() if ln.startswith("|")]
    cells = [[c.strip() for c in ln.strip().strip("|").split("|")] for ln in lines]
    rows = [r for r in cells[1:] if not all(set(c) <= set("-:") for c in r)]
    return (cells[0] if cells else []), rows


def _table(md: str, widths: Sequence[float], size: float = 7.0, drop: int | None = None) -> Block:
    header, rows = md_table(md)
    if drop is not None:
        header = header[:drop] + header[drop + 1 :]
        rows = [r[:drop] + r[drop + 1 :] for r in rows]
    return Block("table", header=header, rows=rows, widths=list(widths), size=size)


def _row(md: str, startswith: str) -> list[str]:
    _, rows = md_table(md)
    return next((r for r in rows if r[0].startswith(startswith)), [])


def self_agreement() -> dict[str, Any] | None:
    """The D4 labeller's v1 labels against their own blind v2 relabel, from the label files."""
    from floorcall.data.d4_llm import agreement
    from floorcall.data.labelling import load_labels

    labels = REPO_ROOT / "data" / "labels"
    v1 = {i: r["label"] for i, r in load_labels(labels / "escalate.labels.jsonl").items()}
    v2 = {i: r["label"] for i, r in load_labels(labels / "escalate.labels.v2.jsonl").items()}
    if not v1 or not v2:
        return None
    return agreement(v2, v1, dict.fromkeys(v2, "all"))


# -- the report ---------------------------------------------------------------------------------


def content(code: str, date: str) -> list[Block]:
    from floorcall import questions, space
    from floorcall.evaluate.paired import BASELINE_COMPARISONS
    from floorcall.evaluate.report import SECTIONS

    ta, tb, pd = RESULTS / "table_a", RESULTS / "table_b", RESULTS / "paired"
    replay = _load(RESULTS / "replay" / "replay.json")
    table_b = SECTIONS["table-b"]()
    blocks = [
        Block("h1", "floorcall"),
        Block("p", "A millisecond decision layer for voice agents. Project report."),
        Block("meta", f"Yash Gudivada · {date} · built from commit {code[:7]}"),
        Block("meta", f"Code: {GITHUB}\nModel: {MODEL}\nDemo: {SPACE}"),
        Block("h2", "Summary"),
        Block(
            "p",
            "A voice agent has to settle a few questions inline, before it acts: has the user "
            "finished, was the speech over the agent a backchannel or a real interruption, what "
            "does the user want, and should a person take over. floorcall answers them together, "
            "from the conversation's text, in one batched forward pass of Laya, a small "
            "open-weight decision encoder, fine-tuned on all four tasks and calibrated per "
            "decision. Every number below is on a frozen test set that no training, calibration "
            "or threshold choice saw.",
        ),
    ]
    best = {d: (a, b) for d, a, b in BASELINE_COMPARISONS}
    items = []
    for d in DECISIONS:
        model = "finetuned_temp_threshold" if d == "escalate" else "finetuned_temp"
        (_, a_model), (_, b_model) = best[d]
        paired = pd / f"{d}.{a_model}_vs_{b_model}.json"
        items.append(
            f"{NAMES[d]}: macro-F1 {_ci(ta / f'{d}.{model}.json', 'macro_f1')}; against "
            f"the best cheap baseline ({BASELINE_NAMES.get(b_model, b_model)}), "
            f"{_diff(paired, 'macro_f1')} paired."
        )
    items.append(
        "Calibration: per-decision temperature takes ECE from "
        + ", ".join(
            f"{_num(ta / f'{d}.finetuned.json', 'metrics', 'ece')} to "
            f"{_num(ta / f'{d}.finetuned_temp.json', 'metrics', 'ece')} ({NAMES[d].split()[0]})"
            for d in DECISIONS
        )
        + "."
    )
    pause = _row(table_b, "GPU, CUDA graphs: user_pause, 3 questions in 1 call")
    items.append(
        "Latency (laptop GPU, batch 1, the pause event's three questions in one call): "
        f"p50 {pause[1] if pause else TODO} ms, p99 {pause[3] if pause else TODO} ms, against a "
        "50 ms p99 budget that no measured path meets."
    )
    td = RESULTS / "paired" / "table_d"
    items.append(
        "Normalization is necessary: a model trained on punctuated text reaches D1 macro-F1 "
        f"{_num(td / 'turn_complete.no_normalize.written_vs_full.json', 'differences', 'macro_f1', 'a')} "
        "on punctuated text and collapses to "
        f"{_num(td / 'turn_complete.no_normalize.asr_vs_full.json', 'differences', 'macro_f1', 'a')} "
        "on ASR-style text."
    )
    if replay:
        (nok, npts), (fok, fpts) = space.score(replay, "naive"), space.score(replay, "floorcall")
        items.append(
            f"Replay demo (illustrative, not an evaluation): a naive agent acted as wanted at "
            f"{nok} of {npts} scripted points, floorcall at {fok} of {fpts}."
        )
    blocks.append(Block("bullets", items=items))

    # -- what it decides
    blocks += [
        Block("h2", "1. The decisions"),
        Block(
            "p",
            "Decisions fire on pipeline events, and each event asks only its own questions, in "
            "one pass. The policy turns the calibrated probabilities into actions through "
            "thresholds chosen on calibration data, each a point on a measured tradeoff curve.",
        ),
        Block(
            "table",
            header=["Event", "Questions in one pass"],
            rows=[[e.value, ", ".join(questions.EVENT_QUESTIONS[e])] for e in questions.Event],
            widths=[60, 114],
            size=8,
        ),
        Block(
            "bullets",
            items=[
                f"{qid}: {q['instructions']} Options: {', '.join(q['criteria'])}."
                if qid != "route"
                else f"{qid}: {q['instructions']} Options: the 15 banking intents of CLINC150 "
                "and out_of_scope."
                for qid, q in questions.ALL_QUESTIONS.items()
            ],
        ),
        Block(
            "p",
            "Why not an LLM: the inline decisions must fit in the tens of milliseconds a voice "
            "agent has left after speech recognition, the response model and speech synthesis. "
            "A prompted LLM called over the network measured p50 "
            f"{_num(tb / 'prompted_llm.json', 'total_ms', 'p50', fmt='{:,.0f}')} ms in Table B.",
        ),
    ]

    # -- data
    rows = []
    for d in DECISIONS:
        card = _load(REPO_ROOT / "data" / "processed" / "cards" / f"{d}.json") or {}
        bal = card.get("balance", {})
        train = bal.get("train", {}).get("rows", TODO)
        if d == "escalate":
            train = _load(RESULTS / "d4_labeller" / "llm-labeller-v3" / "train.llm_v3.json") or {}
            train = train.get("train_rows", TODO)
        test = bal.get("test", {})
        rows.append(
            [
                NAMES[d],
                SOURCE_NAMES.get(str(card.get("source")), TODO),
                str(card.get("licence", TODO)),
                f"{train:,}" if isinstance(train, int) else TODO,
                f"{bal.get('calib', {}).get('rows', 0):,}" if bal else TODO,
                f"{test.get('rows', 0):,}" if test else TODO,
                ", ".join(f"{k} {v:,}" for k, v in test.get("labels", {}).items())
                if d != "route"
                else f"{len(test.get('labels', {}))} labels",
            ]
        )
    agree = self_agreement()
    v3 = RESULTS / "d4_labeller" / "llm-labeller-v3" / "calib.json"
    blocks += [
        Block("h2", "2. Data"),
        Block(
            "table",
            header=["Decision", "Source", "Licence", "Train", "Calib", "Test", "Test balance"],
            rows=rows,
            widths=[24, 18, 24, 15, 13, 13, 67],
            size=7,
        ),
        Block(
            "bullets",
            items=[
                "SwDA is split by conversation, and D1 and D2 share the split, so no test "
                "conversation is seen through either task. Test sets were frozen and hashed "
                "before the first training run, and every result records the hash it was "
                "scored on.",
                "All text, in training and at inference, is normalized as a speech recognizer "
                "delivers it: lowercase, punctuation removed.",
                "D4's test and calib labels are one person's, by hand, on real customer "
                "messages; its training labels come from an LLM (gpt-oss-120b). Relabelling "
                "blind under a revised guideline, the labeller agreed with the first pass at "
                + (
                    f"accuracy {agree['accuracy']:.3f}, Cohen's κ {agree['cohen_kappa']:.3f} "
                    f"(n = {agree['n']})"
                    if agree
                    else TODO
                )
                + "; the LLM labels agreed with the hand labels on calib at κ "
                f"{_num(v3, 'agreement', 'cohen_kappa')}, with escalate recall "
                f"{_num(v3, 'agreement', 'escalate_recall')}.",
                "D4's messages are real tweets, some with personal details, so the public "
                "repository holds their ids, labels and hashes, not their text; `floorcall data "
                "restore-escalate` rebuilds them byte for byte from the corpus.",
            ],
        ),
    ]

    # -- model
    temps = ", ".join(
        f"{NAMES[d].split()[0]} {_num(ta / f'{d}.finetuned_temp.json', 'temperature', fmt='{:.2f}')}"
        for d in DECISIONS
    )
    blocks += [
        Block("h2", "3. Model and training"),
        Block(
            "p",
            f"Base model: convaiinnovations/laya (English), revision "
            f"{(LayaSettings().revision or TODO)[:7]}, a ModernBERT-large encoder with a decision "
            "head. One multi-task checkpoint is fine-tuned on all four decisions with soft "
            "cross-entropy; the kept epoch is the one with the lowest cross-entropy on a dev "
            "split held out of train by conversation. Temperatures are then fitted per "
            f"decision on calib ({temps}), and D4's decision threshold is chosen on calib by "
            "macro-F1. The fine-tuned rows below are this checkpoint, released as "
            f"{MODEL.split('co/')[1]} under CC BY-NC-SA 4.0.",
        ),
    ]

    # -- results
    table_a = SECTIONS["table-a"]()
    blocks.append(Block("h2", "4. Results"))
    blocks.append(Block("p", "Table A: quality per decision. Brackets are 95% intervals."))
    header, rows = md_table(table_a)
    for d in DECISIONS:
        sub = [r[1:] for r in rows if r[0] == NAMES[d]]
        blocks.append(Block("meta", NAMES[d]))
        blocks.append(
            Block("table", header=header[1:], rows=sub, widths=[50, 26, 24, 24, 24, 26], size=6.8)
        )
    blocks += [
        Block(
            "p", "Table B: latency at batch 1. Budgets: p99 at most 50 ms on GPU, 100 ms on CPU."
        ),
        _table(table_b, [62, 16, 16, 16, 20, 20, 24], size=6.5),
        Block("meta", SECTIONS["table-b-env"]()),
        Block("p", "Table C: robustness to ASR-style noise. Cells are macro-F1 (accuracy)."),
        _table(SECTIONS["table-c"](), [38, 34, 34, 34, 34], size=7),
        Block(
            "p",
            "Table D: ablations. Each arm is trained again without part of the state, on "
            "Kaggle, and compared with a full arm trained the same way there. Each arm is one "
            "run; differences smaller than the gap between two runs of one recipe are not read "
            "as effects.",
        ),
        _table(SECTIONS["table-d"](), [44, 22, 18, 22, 18, 25, 25], size=6.5),
        Block("p", "Each variant minus the full arm, with paired-bootstrap 95% intervals:"),
        _table(SECTIONS["table-d-paired"](), [36, 23, 23, 23, 23, 23, 23], size=6),
        Block("p", "Operating points, each threshold chosen on calib:"),
        _table(SECTIONS["operating-points"](), [30, 18, 26, 28, 18, 28, 26], size=6.8),
    ]
    curves = _load(RESULTS / "curves" / "finetuned_temp.json") or {}
    blocks.append(
        Block(
            "meta",
            f"The curves and reliability figures are computed at {curves.get('precision', TODO)} "
            "inference, the default since D-040; Table A's fine-tuned rows were scored before "
            "that change, so a figure's ECE can differ from Table A's in the third decimal.",
        )
    )
    for name, caption in (
        ("tradeoff_interrupt", "θ_interrupt: false stops against missed interruptions"),
        ("tradeoff_yield", "θ_yield: premature responses against added delay"),
        ("reliability_finetuned_temp", "Reliability, fine-tuned, before and after temperature"),
    ):
        path = RESULTS / "figures" / f"{name}.light.png"
        blocks.append(Block("image", caption, path=path) if path.exists() else Block("p", TODO))

    # -- replay
    if replay:
        misses = []
        for s in replay["scripts"]:
            for o in s["runs"]["floorcall"]["outcomes"]:
                if o["ok"] is False:
                    misses.append(
                        f"{s['id']}: wanted {o['want'].replace('_', ' ')}"
                        + (f", route {o['want_route']}" if o.get("want_route") else "")
                        + (", hand off" if o.get("want_escalate") else "")
                        + f"; floorcall: {o['act']['action'].replace('_', ' ')}"
                        + (f", route {o['act']['route']}" if o["act"].get("route") else "")
                        + f" ({o['act']['reason']})."
                    )
        blocks += [
            Block("h2", "5. Replay: the same call through two agents"),
            Block(
                "p",
                "Eight scripted banking calls run through a naive agent, which answers after "
                f"{replay['naive_silence_ms']} ms of silence and stops for any speech, and "
                "through floorcall. The scripts were frozen by hash before floorcall first ran on "
                f"them. The naive agent acted as wanted at {nok} of {npts} points and floorcall "
                f"at {fok} of {fpts}. floorcall's misses:",
            ),
            Block("bullets", items=misses),
        ]

    # -- limitations
    lex = ta / "barge_in.lexical_rule.json"
    ft2 = ta / "barge_in.finetuned_temp.json"
    c = SECTIONS["table-c"]()
    d2 = _row(c, "D2 barge_in")
    blocks += [
        Block("h2", "6. Limitations"),
        Block(
            "bullets",
            items=[
                "Latency: no measured GPU path meets p99 ≤ 50 ms on the laptop GPU, and every CPU "
                "path misses p99 ≤ 100 ms by a wide margin (Table B).",
                f"D2: a lexical rule on backchannel words and length reaches macro-F1 "
                f"{_num(lex, 'metrics', 'macro_f1')} against the model's "
                f"{_num(ft2, 'metrics', 'macro_f1')}, and is better on D2's hard subset "
                f"({_num(lex, 'metrics', 'hard_accuracy')} against "
                f"{_num(ft2, 'metrics', 'hard_accuracy')}).",
                "D4: its labels have a low ceiling (section 2), its test set is 200 messages, and "
                "at its calib threshold it ties TF-IDF with logistic regression on macro-F1 "
                f"({_diff(pd / 'escalate.finetuned_temp_threshold_vs_tfidf_lr.json', 'macro_f1')}"
                "); it is better calibrated.",
                "D1 reads text only and cannot hear intonation; SwDA is two people on the phone "
                "from 1990 to 1991, not a user talking to an agent. Its hard subset, truncations ending "
                "on a turn-final word, is right at "
                f"{_num(ta / 'turn_complete.finetuned_temp.json', 'metrics', 'hard_accuracy')}.",
                "ASR noise costs accuracy everywhere (Table C): D2 falls from "
                f"{d2[1].split()[0] if d2 else TODO} to {d2[4].split()[0] if d2 else TODO} "
                "macro-F1 at the heaviest noise level.",
                "Table D's arms are single runs, so its hard-subset movements cannot be "
                "separated from run-to-run variation.",
            ],
        ),
    ]

    # -- reproduce
    blocks += [
        Block("h2", "7. Reproducing it"),
        Block(
            "code",
            f"git clone {GITHUB}\ncd floorcall\nuv sync                        # NVIDIA GPU\n"
            "uv run pytest\nuv run floorcall data restore-escalate   # D4 text, from the corpus\n"
            "uv run floorcall replay demo/scripts/*.json --compare naive\n"
            "uv run floorcall eval readme   # every table, from results/\n"
            "uv run floorcall release report",
        ),
        Block(
            "bullets",
            items=[
                "Every result file records the commit that produced it. The public history was "
                "rewritten once to remove D4's tweet text; docs/commit-map.txt maps the commits "
                "cited by results files to their public SHAs.",
                "Every divergence from the original plan, and why, is in docs/DECISIONS.md.",
                "Licences: code Apache-2.0; the weights CC BY-NC-SA 4.0; data files carry their "
                "sources' licences (data/LICENSE.md): SwDA CC BY-NC-SA 3.0, CLINC150 CC BY 3.0, "
                "Customer Support on Twitter CC BY-NC-SA 4.0.",
            ],
        ),
    ]
    return blocks


# -- drawing ---------------------------------------------------------------------------------


def _fonts() -> Path:
    import matplotlib

    return Path(matplotlib.get_data_path()) / "fonts" / "ttf"


def render(blocks: Sequence[Block], out: Path, created: datetime) -> int:
    """Draw the blocks into `out`; returns the page count."""
    from fpdf import FPDF
    from fpdf.fonts import FontFace

    class Report(FPDF):
        def footer(self) -> None:
            self.set_y(-12)
            self.set_font("dejavu", "", 7)
            self.set_text_color(110)
            self.cell(0, 6, f"floorcall · project report · page {self.page_no()}", align="C")

    pdf = Report(format="A4")
    pdf.set_creation_date(created)
    pdf.set_title("floorcall: project report")
    pdf.set_author("Yash Gudivada")
    pdf.set_margins(18, 16, 18)
    pdf.set_auto_page_break(True, margin=16)
    fonts = _fonts()
    pdf.add_font("dejavu", "", fonts / "DejaVuSans.ttf")
    pdf.add_font("dejavu", "B", fonts / "DejaVuSans-Bold.ttf")
    pdf.add_font("dejavu", "I", fonts / "DejaVuSans-Oblique.ttf")
    pdf.add_font("mono", "", fonts / "DejaVuSansMono.ttf")
    pdf.add_page()
    width = pdf.epw
    for b in blocks:
        pdf.set_text_color(25)
        if b.kind == "h1":
            pdf.set_font("dejavu", "B", 22)
            pdf.cell(0, 11, b.text, new_x="LMARGIN", new_y="NEXT")
        elif b.kind == "h2":
            pdf.ln(3)
            pdf.set_font("dejavu", "B", 12.5)
            pdf.cell(0, 8, b.text, new_x="LMARGIN", new_y="NEXT")
        elif b.kind == "p":
            pdf.set_font("dejavu", "", 9)
            pdf.multi_cell(width, 4.6, b.text, new_x="LMARGIN", new_y="NEXT")
            pdf.ln(1.5)
        elif b.kind == "meta":
            pdf.set_font("dejavu", "I", 7.5)
            pdf.set_text_color(90)
            pdf.multi_cell(width, 3.8, b.text, align="L", new_x="LMARGIN", new_y="NEXT")
            pdf.ln(1)
        elif b.kind == "bullets":
            pdf.set_font("dejavu", "", 9)
            for item in b.items:
                pdf.set_x(pdf.l_margin)
                pdf.cell(5, 4.6, "•")
                pdf.multi_cell(width - 5, 4.6, item, new_x="LMARGIN", new_y="NEXT")
                pdf.ln(0.8)
            pdf.ln(1)
        elif b.kind == "code":
            pdf.set_font("mono", "", 7.8)
            pdf.set_fill_color(240, 242, 245)
            pdf.multi_cell(width, 4.2, b.text, fill=True, new_x="LMARGIN", new_y="NEXT")
            pdf.ln(2)
        elif b.kind == "table":
            pdf.set_font("dejavu", "", b.size)
            scale = width / sum(b.widths)
            with pdf.table(
                col_widths=[w * scale for w in b.widths],
                width=width,
                text_align="LEFT",
                line_height=b.size * 0.5,
                headings_style=FontFace(emphasis="BOLD", fill_color=(232, 236, 241)),
                borders_layout="HORIZONTAL_LINES",
                padding=0.8,
            ) as table:
                for cells in [b.header, *b.rows]:
                    row = table.row()
                    for cell in cells:
                        row.cell(cell)
            pdf.ln(3)
        elif b.kind == "image" and b.path is not None:
            img_w = min(width, 150.0)
            pdf.image(str(b.path), x=pdf.l_margin + (width - img_w) / 2, w=img_w)
            pdf.set_font("dejavu", "I", 7.5)
            pdf.cell(0, 5, b.text, align="C", new_x="LMARGIN", new_y="NEXT")
            pdf.ln(2)
    out.parent.mkdir(parents=True, exist_ok=True)
    pdf.output(str(out))
    return pdf.page_no()


def build(out: Path = REPORT, *, allow_dirty: bool = False) -> tuple[Path, int, str]:
    """Build the report from the checkout's HEAD; refuses a dirty tree unless told otherwise."""
    import subprocess

    def git(*args: str) -> str:
        return subprocess.run(
            ["git", *args], cwd=REPO_ROOT, capture_output=True, text=True, check=True
        ).stdout.strip()

    if not allow_dirty and git("status", "--porcelain"):
        raise ValueError("the working tree is not clean: the report would not match its commit")
    sha = git("rev-parse", "HEAD")
    created = datetime.fromisoformat(git("log", "-1", "--format=%cI"))
    pages = render(content(sha, created.date().isoformat()), out, created)
    return out, pages, sha
