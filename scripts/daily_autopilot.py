"""Self-driving daily "autopilot" for the ARC-AGI-2 Kaggle submission.

The competition scores one committed notebook version per day (a code
competition: `kaggle competitions submit -k <kernel> -v <version>` only works
against a kernel version that has already been pushed AND finished running).
Kaggle GPU kernels take hours, so improving the live submission is an async
pipeline: train an adapter -> measure it on the public-eval canary -> only if
it beats the current best, commit-run the real submission notebook -> submit
-> wait for the scored rerun to land on the leaderboard -> keep or roll back.

This script advances that pipeline by one **tick**. A scheduler (see
`scripts/install_autopilot.sh`, a macOS launchd agent) fires
`python scripts/daily_autopilot.py --once` a few times a day; each invocation
loads `artifacts/autopilot_state.json`, advances the state machine by exactly
one step, and saves it back. Every tick is idempotent and safe to re-run: a
missed or duplicate fire just resumes/no-ops off the persisted state.

State machine (see `tick()`):
    1. Reconcile `pending_lb` — if a previous submission now has a public
       score, promote `live_config` when it's an improvement, else reject it.
    2. If a kernel is `inflight`, poll its status:
         - RUNNING/QUEUED (or anything unrecognized)  -> wait.
         - ERROR/CANCEL                                -> log + clear, no retry.
         - COMPLETE -> dispatch by `kind`:
             * train         -> stage the trained adapter as a Kaggle Dataset,
                                 set `candidate` (awaiting an eval gate).
             * eval          -> gate on `correct >= best_eval_correct + 1`;
                                 pass -> `gated_candidate`, fail -> discard.
             * submit_commit -> call `competition_submit`, set `pending_lb`.
    3. Else, if a candidate passed the eval gate and we haven't submitted
       today -> push the real submission notebook (commit run).
    4. Else, if a trained-but-unevaluated candidate exists -> push an eval
       kernel to measure it (spends GPU, zero submissions).
    5. Else, if under the weekly GPU-hour quota -> launch the next backlog
       experiment (train or eval), round-robin.
    6. Always append a one-line dated status to `SUBMISSION_LOG.md` and print it.

Every Kaggle-touching call goes through `KaggleClient`, a thin subprocess
wrapper — every method is a plain function that tests replace with a fake, so
the whole state machine is exercised with zero network calls. Kaggle facts
this module encodes (see `DEPLOYMENT.md` Sec 3 and `.remember/` for the full
war-story): `--accelerator NvidiaTeslaT4` is the only push-time value that
reliably works; never write `machine_shape`/`docker_image` into a pushed
`kernel-metadata.json`; a push can come back BUSY ("Maximum batch GPU session
count of 2 reached") rather than erroring, which this module treats as
"nothing to do this tick, retry later" rather than a failure.

Notebook/kernel-folder construction and the experiment backlog live in
`autopilot_kernels.py`; shared constants (slugs, mount paths, quotas) live in
`autopilot_config.py` — split out purely to keep each file under the repo's
~400-800 line guideline.
"""

from __future__ import annotations

import argparse
import ast
import json
import logging
import re
import subprocess
import sys
from dataclasses import dataclass
from datetime import date
from pathlib import Path

# scripts/ has no __init__.py (plain script directory, matching kaggle_eval.py's
# `from kaggle_submit import ...` convention) — callers (the CLI entrypoint, or
# a test loading this file via importlib) are responsible for putting scripts/
# on sys.path before importing this module. `__main__` below adds it too, so
# `python scripts/daily_autopilot.py` works standalone.
from autopilot_config import (
    ADAPTER_DATASET_SLUG,
    ADAPTER_MOUNT,
    BASE_MODEL_MOUNT,
    COMPETITION_SLUG,
    EXPLORE_COMMIT_HOURS,
    GPU_ACCELERATOR,
    KERNEL_RUNS_DIR,
    STATE_PATH,
    SUBMISSION_KERNEL,  # noqa: F401 — base id, re-exported for tests/operators
    SUBMISSION_LOG_PATH,
    TRAIN_KERNEL,
    WEEKLY_GPU_HOUR_QUOTA,
)
from autopilot_kernels import (
    BACKLOG,
    dataset_sources_for,
    eval_run_src,
    exploration_variants,
    model_sources_for,
    stage_dir,
    submission_kernel_for,
    submission_run_src,
    write_dataset_metadata,
    write_submission_kernel,
)

logger = logging.getLogger("arc.autopilot")


# ---------------------------------------------------------------------------
# KaggleClient — thin, fully-mockable subprocess wrapper over the `kaggle` CLI
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class PushResult:
    """Outcome of `kaggle kernels push`."""

    ok: bool
    busy: bool
    version: int | None
    error: str | None


@dataclass(frozen=True)
class Submission:
    """One row of `kaggle competitions submissions`."""

    ref: str
    date: str
    description: str
    public_score: str | None
    status: str


def _safe_json_list(text: str) -> list:
    """Parse `text` as a JSON list, defensively — CLI output can carry stray
    warnings before/after the JSON payload, so fall back to extracting the
    first `[...]` span before giving up (empty list, never raises)."""
    try:
        data = json.loads(text)
        return data if isinstance(data, list) else []
    except (json.JSONDecodeError, TypeError):
        pass
    match = re.search(r"\[.*\]", text, re.DOTALL)
    if not match:
        return []
    try:
        data = json.loads(match.group(0))
        return data if isinstance(data, list) else []
    except json.JSONDecodeError:
        return []


class KaggleClient:
    """Every Kaggle interaction the autopilot needs, via the `kaggle` CLI.

    Parsing is deliberately defensive (regex/substring match on CLI text,
    never an assumption about exact formatting) since the CLI's output is not
    a stable contract. Every method here has a same-shaped fake used by
    `tests/test_autopilot.py` — no method should grow logic that can't be
    exercised without a real Kaggle account.
    """

    def __init__(self, competition: str = COMPETITION_SLUG, timeout_s: float = 180.0):
        self.competition = competition
        self.timeout_s = timeout_s

    def _run(self, args: list[str]) -> subprocess.CompletedProcess:
        cmd = ["kaggle", *args]
        logger.info("kaggle CLI: %s", " ".join(cmd))
        return subprocess.run(  # noqa: S603 — fixed "kaggle" executable, args are ours
            cmd, capture_output=True, text=True, timeout=self.timeout_s, check=False
        )

    def kernel_status(self, kernel: str) -> str:
        """One of COMPLETE/ERROR/CANCEL/RUNNING/QUEUED/UNKNOWN."""
        proc = self._run(["kernels", "status", kernel])
        text = f"{proc.stdout}\n{proc.stderr}".upper()
        # A 404 is a PHANTOM (no session found: transient right after a push,
        # and — since Kaggle's ~2026-07-14 API change — for kernels with no
        # active session), NOT a run failure. It must be detected BEFORE the
        # token scan: "404 CLIENT ERROR" contains the substring "ERROR", and
        # misreading it once abandoned healthy runs on the spot. UNKNOWN flows
        # into the >=STALE_INFLIGHT_DAYS guard instead.
        if "404" in text or "NOT FOUND" in text:
            return "UNKNOWN"
        for token in ("COMPLETE", "ERROR", "CANCEL", "RUNNING", "QUEUED"):
            if token in text:
                return token
        return "UNKNOWN"

    def kernel_push(self, folder, accelerator: str = GPU_ACCELERATOR) -> PushResult:
        proc = self._run(["kernels", "push", "-p", str(folder), "--accelerator", accelerator])
        text = f"{proc.stdout}\n{proc.stderr}"
        if re.search(r"maximum batch gpu session", text, re.IGNORECASE):
            # Not an error: both GPU push slots are busy. Retry next tick.
            return PushResult(ok=False, busy=True, version=None, error=text.strip())
        match = re.search(r"[Kk]ernel version (\d+) successfully pushed", text)
        if match:
            return PushResult(ok=True, busy=False, version=int(match.group(1)), error=None)
        if re.search(r"kernel push error", text, re.IGNORECASE) or proc.returncode != 0:
            return PushResult(
                ok=False,
                busy=False,
                version=None,
                error=text.strip() or f"push failed (rc={proc.returncode})",
            )
        return PushResult(
            ok=False,
            busy=False,
            version=None,
            error=f"unrecognized push output: {text.strip()[:300]}",
        )

    def kernel_output(self, kernel: str, dest_dir) -> Path:
        dest = Path(dest_dir)
        dest.mkdir(parents=True, exist_ok=True)
        self._run(["kernels", "output", kernel, "-p", str(dest)])
        return dest

    def dataset_create_or_version(self, folder) -> bool:
        """Version the dataset if it exists, else create it. Returns success."""
        folder = Path(folder)
        version_proc = self._run(
            [
                "datasets", "version", "-p", str(folder),
                "-m", "autopilot: staged update", "-r", "zip",
            ]
        )
        text = f"{version_proc.stdout}\n{version_proc.stderr}"
        if version_proc.returncode == 0 and not re.search(r"error", text, re.IGNORECASE):
            return True
        create_proc = self._run(["datasets", "create", "-p", str(folder), "-r", "zip"])
        text2 = f"{create_proc.stdout}\n{create_proc.stderr}"
        return create_proc.returncode == 0 and not re.search(r"error", text2, re.IGNORECASE)

    def competition_submit(self, kernel: str, version: int, message: str) -> bool:
        # `-f submission.json` names the kernel's output file. Optional under
        # the old API, REQUIRED since Kaggle's ~2026-07-14 migration — without
        # it CreateCodeSubmission returns a bare "400 Bad Request" (this burned
        # three exploration submissions before the cause was found).
        proc = self._run(
            [
                "competitions", "submit", self.competition,
                "-k", kernel, "-v", str(version),
                "-f", "submission.json", "-m", message,
            ]
        )
        text = f"{proc.stdout}\n{proc.stderr}"
        ok = proc.returncode == 0 and not re.search(r"error", text, re.IGNORECASE)
        if not ok:
            # Surface the CLI's actual complaint — a bool alone cost a day of
            # diagnosis per failure when the API contract changed.
            logger.warning("competition_submit failed: %s", text.strip()[:500])
        return ok

    def competition_submissions(self) -> list[Submission]:
        proc = self._run(["competitions", "submissions", self.competition, "--format", "json"])
        rows = _safe_json_list(proc.stdout)
        out = []
        for row in rows:
            if not isinstance(row, dict):
                continue
            out.append(
                Submission(
                    ref=str(row.get("ref", row.get("id", row.get("submissionId", "")))),
                    date=str(row.get("date", row.get("submittedDate", ""))),
                    description=str(row.get("description", "")),
                    public_score=row.get("publicScore", row.get("public_score")),
                    status=str(row.get("status", "")),
                )
            )
        return out


class DryRunKaggleClient(KaggleClient):
    """A KaggleClient that only logs — never shells out. Wired up by
    `--dry-run` so an operator can rehearse the state machine without
    touching Kaggle (every push comes back "not pushed", so no inflight
    kernel is ever created — safe to run any time)."""

    def kernel_status(self, kernel: str) -> str:
        logger.info("[dry-run] kernel_status(%s)", kernel)
        return "UNKNOWN"

    def kernel_push(self, folder, accelerator: str = GPU_ACCELERATOR) -> PushResult:
        logger.info("[dry-run] kernel_push(%s, accelerator=%s)", folder, accelerator)
        return PushResult(ok=False, busy=False, version=None, error="dry-run: not pushed")

    def kernel_output(self, kernel: str, dest_dir) -> Path:
        logger.info("[dry-run] kernel_output(%s -> %s)", kernel, dest_dir)
        return Path(dest_dir)

    def dataset_create_or_version(self, folder) -> bool:
        logger.info("[dry-run] dataset_create_or_version(%s)", folder)
        return False

    def competition_submit(self, kernel: str, version, message: str) -> bool:
        logger.info("[dry-run] competition_submit(%s, v%s, %r)", kernel, version, message)
        return False

    def competition_submissions(self) -> list[Submission]:
        logger.info("[dry-run] competition_submissions()")
        return []


# ---------------------------------------------------------------------------
# State — a plain dict persisted as JSON; see the module docstring for shape.
# ---------------------------------------------------------------------------


def default_state() -> dict:
    """Sane defaults for a first run (no `artifacts/autopilot_state.json` yet)."""
    return {
        "best_eval_correct": 0,
        "best_lb": 0.83,
        "live_config": {
            "model_path": BASE_MODEL_MOUNT,
            "adapter_path": None,
            "llm_kwargs": {"selection": "votes"},
            "ttt_config": None,
        },
        "inflight": None,
        "candidate": None,
        "gated_candidate": None,
        "pending_lb": None,
        "last_submit_date": "",
        "gpu_hours_week": 0.0,
        "week_start": "",
        "backlog_index": 0,
        "push_failures": 0,
        "explore_inflight": None,
        "explored": [],
        "extra_exploration_variants": [],
        # Rotating submission-kernel id (Kaggle busy-create burn bug — see
        # autopilot_kernels.submission_kernel_for): seq picks the id, created
        # pins it once a push has succeeded (existing kernels update fine).
        "submission_kernel_seq": 0,
        "submission_kernel_created": False,
    }


def load_state(path: Path | None = None) -> dict:
    """Load state, falling back to defaults on a missing/corrupt file. Any
    key absent from the persisted JSON (e.g. after adding a new field) is
    backfilled from `default_state()`.

    `path` defaults to the module-level `STATE_PATH` — resolved at CALL time
    (not baked into a default-argument value at def time) so tests can
    `monkeypatch.setattr(module, "STATE_PATH", tmp_path / ...)` and have
    every no-arg caller (including `main()`) honor it.
    """
    path = path if path is not None else STATE_PATH
    state = default_state()
    if not path.exists():
        return state
    try:
        loaded = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        logger.warning("could not read state at %s (%s); starting fresh", path, exc)
        return state
    if isinstance(loaded, dict):
        state.update(loaded)
    return state


def save_state(state: dict, path: Path | None = None) -> None:
    """Persist `state` as JSON. `path` defaults to `STATE_PATH`, resolved at
    call time — see `load_state` for why that matters for monkeypatching."""
    path = path if path is not None else STATE_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(state, indent=2, sort_keys=True), encoding="utf-8")


# ---------------------------------------------------------------------------
# SUBMISSION_LOG.md — append-only human-readable record
# ---------------------------------------------------------------------------

_LOG_HEADER = (
    "# Submission Log\n\n"
    "Chronological record of every Kaggle competition submission (public-LB "
    "reconciliation) and autopilot tick status, appended by "
    "`scripts/daily_autopilot.py`.\n\n"
    "| Date | Ref | Config | LB Score | Note |\n"
    "|------|-----|--------|----------|------|\n"
)


def _ensure_submission_log() -> None:
    if not SUBMISSION_LOG_PATH.exists():
        SUBMISSION_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
        SUBMISSION_LOG_PATH.write_text(_LOG_HEADER, encoding="utf-8")


def _append_submission_log_row(today: str, ref: str, label: str, score, note: str) -> None:
    _ensure_submission_log()
    with SUBMISSION_LOG_PATH.open("a", encoding="utf-8") as f:
        f.write(f"| {today} | {ref} | {label} | {score} | {note} |\n")


def _log_status(today: str, message: str) -> None:
    logger.info(message)
    _append_submission_log_row(today, "-", "-", "-", message)
    print(f"[{today}] {message}")


def _summarize_config(config: dict) -> str:
    """Compact one-line label for a config dict, for SUBMISSION_LOG.md rows."""
    has_adapter = config.get("adapter_dataset") or config.get("adapter_path")
    adapter = config.get("adapter_dataset") or ("adapter" if has_adapter else "no-adapter")
    selection = (config.get("llm_kwargs") or {}).get("selection", "votes")
    parts = [str(adapter), f"selection={selection}"]
    ttt_steps = (config.get("ttt_config") or {}).get("max_steps")
    if ttt_steps:
        parts.append(f"ttt_max_steps={ttt_steps}")
    return " ".join(parts)


# ---------------------------------------------------------------------------
# tick() and its per-branch helpers
# ---------------------------------------------------------------------------


def _find_first(root: Path, filename: str) -> Path | None:
    if not root.exists():
        return None
    matches = sorted(root.rglob(filename))
    return matches[0] if matches else None


_SUMMARY_RE = re.compile(r"summary:\s*(\{.*\})")


def _find_summary(root: Path) -> dict | None:
    """Scan every downloaded output file for a `summary: {...}` line (printed
    by `kaggle_eval.main`) and parse it with `ast.literal_eval` (it's a
    repr'd Python dict, not JSON — single-quoted keys)."""
    if not root.exists():
        return None
    for path in sorted(root.rglob("*")):
        if not path.is_file():
            continue
        try:
            text = path.read_text(encoding="utf-8", errors="ignore")
        except OSError:
            continue
        match = _SUMMARY_RE.search(text)
        if not match:
            continue
        try:
            parsed = ast.literal_eval(match.group(1))
        except (ValueError, SyntaxError):
            continue
        if isinstance(parsed, dict):
            return parsed
    return None


def _safe_name(kernel: str) -> str:
    return kernel.replace("/", "_")


def _parse_score(raw) -> float | None:
    if raw in (None, "", "None", "N/A", "pending"):
        return None
    try:
        return float(raw)
    except (TypeError, ValueError):
        return None


def _find_submission(subs: list[Submission], ref: str) -> Submission | None:
    for s in subs:
        if s.ref == ref or (s.description and ref in s.description):
            return s
    return None


def _ensure_week_start(state: dict, today: str) -> dict:
    start = state.get("week_start")
    if not start:
        return {**state, "week_start": today}
    if (date.fromisoformat(today) - date.fromisoformat(start)).days >= 7:
        return {**state, "week_start": today, "gpu_hours_week": 0.0}
    return state


def _reconcile_pending_lb(
    client: KaggleClient, state: dict, today: str
) -> tuple[dict, str | None]:
    pending = state.get("pending_lb")
    if not pending:
        return state, None
    match = _find_submission(client.competition_submissions(), pending["ref"])
    if match is None:
        return state, f"pending_lb: awaiting a scored submission for ref={pending['ref']}"
    score = _parse_score(match.public_score)
    if score is None:
        return state, f"pending_lb: submission {pending['ref']} has no numeric score yet"
    promoted = score >= state["best_lb"]
    new_state = dict(state)
    note = "promoted" if promoted else "rejected"
    if promoted:
        new_state["best_lb"] = score
        new_state["live_config"] = pending["config"]
    new_state["pending_lb"] = None
    label = _summarize_config(pending["config"])
    _append_submission_log_row(today, pending["ref"], label, score, note)
    return new_state, f"pending_lb resolved: ref={pending['ref']} score={score} -> {note}"


def _finish_train(client: KaggleClient, state: dict, inflight: dict) -> tuple[dict, str]:
    kernel_dir = _safe_name(inflight["kernel"])
    dest = KERNEL_RUNS_DIR / kernel_dir / f"v{inflight['version']}" / "train"
    out_dir = client.kernel_output(inflight["kernel"], dest)
    adapter_file = _find_first(out_dir, "adapter_model.safetensors")
    cleared = {**state, "inflight": None}
    if adapter_file is None:
        return cleared, "train COMPLETE but produced no adapter"
    adapter_folder = adapter_file.parent
    write_dataset_metadata(adapter_folder, ADAPTER_DATASET_SLUG, "ARC-AGI-2 autopilot adapter")
    if not client.dataset_create_or_version(adapter_folder):
        return cleared, "train COMPLETE but adapter dataset staging FAILED"
    candidate = {
        "config": {
            **inflight["config"],
            "adapter_path": ADAPTER_MOUNT,
            "adapter_dataset": ADAPTER_DATASET_SLUG,
        },
        "adapter_dataset": ADAPTER_DATASET_SLUG,
    }
    msg = f"train COMPLETE: adapter staged -> {ADAPTER_DATASET_SLUG}; candidate ready for eval"
    return {**cleared, "candidate": candidate}, msg


def _finish_eval(client: KaggleClient, state: dict, inflight: dict) -> tuple[dict, str]:
    kernel_dir = _safe_name(inflight["kernel"])
    dest = KERNEL_RUNS_DIR / kernel_dir / f"v{inflight['version']}" / "eval"
    out_dir = client.kernel_output(inflight["kernel"], dest)
    summary = _find_summary(out_dir)
    candidate = state.get("candidate")
    base_state = {**state, "inflight": None, "candidate": None}
    if not summary or "correct" not in summary:
        return base_state, "eval COMPLETE but no summary parsed; discarding candidate"
    correct = int(summary["correct"])
    best = int(state.get("best_eval_correct", 0))
    if correct >= best + 1:
        gated = {
            "config": (candidate or {}).get("config", inflight["config"]),
            "version": inflight["version"],
            "eval_correct": correct,
        }
        msg = f"eval COMPLETE: correct={correct} >= best({best})+1 -> gated for submit"
        return {**base_state, "gated_candidate": gated}, msg
    msg = f"eval COMPLETE: no improvement ({correct} vs best {best}); discarded"
    # A gate-failed TRAINED-ADAPTER candidate still gets its lottery ticket:
    # at <1% ability the canary can't see sub-1/120 gains, and Kaggle ranks the
    # BEST submission, so queueing it for an exploration submission is free
    # upside. Only adapter candidates qualify — config tweaks are already in
    # the static exploration rotation.
    cand_config = (candidate or {}).get("config")
    has_adapter = bool(
        cand_config
        and (cand_config.get("adapter_dataset") or (candidate or {}).get("adapter_dataset"))
    )
    if has_adapter:
        extra = list(state.get("extra_exploration_variants", []))
        name = f"adapter-eval-v{inflight['version']}"
        if all(v.get("name") != name for v in extra):
            extra.append({"name": name, "config": cand_config})
            base_state = {**base_state, "extra_exploration_variants": extra}
            msg += "; queued for exploration submission"
    return base_state, msg


def _finish_submit_commit(
    client: KaggleClient, state: dict, today: str, inflight: dict
) -> tuple[dict, str]:
    gated = state.get("gated_candidate")
    cleared = {**state, "inflight": None}
    if gated is None:
        return cleared, "submit_commit COMPLETE but no gated_candidate; nothing to submit"
    ref = f"{inflight['kernel']}:v{inflight['version']}:{today}"
    message = f"autopilot {ref} correct={gated.get('eval_correct')}"
    if not client.competition_submit(inflight["kernel"], inflight["version"], message):
        return cleared, f"competition_submit FAILED for v{inflight['version']}"
    new_state = {
        **cleared,
        "gated_candidate": None,
        "pending_lb": {
            "ref": ref,
            "config": gated["config"],
            "eval_correct": gated.get("eval_correct"),
        },
        "last_submit_date": today,
        "best_eval_correct": gated.get("eval_correct", state.get("best_eval_correct", 0)),
    }
    return new_state, f"SUBMITTED: {ref} ({message!r})"


# A waiting kernel unresolved for this many calendar days is treated as dead
# (any real T4 run completes in < 12 h) and abandoned so the loop never wedges.
STALE_INFLIGHT_DAYS = 2


def _days_between(start_iso: str, end_iso: str) -> int:
    """Whole days from `start_iso` to `end_iso` (both ISO dates); 0 on parse error."""
    try:
        return (date.fromisoformat(end_iso) - date.fromisoformat(start_iso)).days
    except (ValueError, TypeError):
        return 0


def _dispatch_inflight(client: KaggleClient, state: dict, today: str) -> tuple[dict, str]:
    inflight = state["inflight"]
    status = client.kernel_status(inflight["kernel"]).upper()
    label = f"{inflight['kind']} kernel {inflight['kernel']} v{inflight['version']}"
    if status == "COMPLETE":
        if inflight["kind"] == "train":
            return _finish_train(client, state, inflight)
        if inflight["kind"] == "eval":
            return _finish_eval(client, state, inflight)
        if inflight["kind"] == "submit_commit":
            return _finish_submit_commit(client, state, today, inflight)
        return {**state, "inflight": None}, f"unknown inflight kind={inflight['kind']!r}; clearing"
    if status in ("ERROR", "CANCEL"):
        # Simple per-purpose failure: log and clear, no automatic retry loop.
        return {**state, "inflight": None}, f"FAILED: {label} status={status}"
    # Waiting (RUNNING/QUEUED/UNKNOWN). Guard the UNATTENDED case: a phantom
    # kernel that never resolves (e.g. Kaggle's session-status endpoint stuck on
    # 404, or a run that silently died) must not wedge the autopilot forever.
    # Stamp the first-observed date, then abandon after STALE_INFLIGHT_DAYS
    # (any real T4 run finishes < 12 h, so >=2 calendar days means it's dead).
    since = inflight.get("since", today)
    if _days_between(since, today) >= STALE_INFLIGHT_DAYS:
        return {**state, "inflight": None}, (
            f"STALE (>={STALE_INFLIGHT_DAYS}d unresolved) {label} status={status}; abandoning"
        )
    return {**state, "inflight": {**inflight, "since": since}}, f"waiting: {label} status={status}"


def _launch_submit_commit(client: KaggleClient, state: dict) -> tuple[dict, str]:
    # SUBMISSION_KERNEL must only ever be held by ONE lane: kernel_status polls
    # per kernel (not per version), so two concurrent sessions on it would
    # conflate statuses between the lanes. Explore commits are short (~1.5 h);
    # defer one tick.
    if state.get("explore_inflight"):
        return state, "submit_commit deferred: exploration commit holds the submission kernel"
    gated = state["gated_candidate"]
    kernel_id = submission_kernel_for(state)
    folder = stage_dir("submit")
    run_src = submission_run_src(gated["config"])
    write_submission_kernel(
        folder, run_src,
        dataset_sources=dataset_sources_for(gated["config"]),
        model_sources=model_sources_for(gated["config"]),
        kernel_id=kernel_id,
    )
    push = client.kernel_push(folder, accelerator=GPU_ACCELERATOR)
    if rotated := _rotate_burned_submission_kernel(state, push):
        return rotated
    if push.busy:
        return state, "submit push BUSY (2 GPU slots in use); retrying next tick"
    if not push.ok:
        return state, f"submit push FAILED: {push.error}"
    inflight = {
        "kernel": kernel_id,
        "version": push.version,
        "kind": "submit_commit",
        "purpose": "commit-run the gated candidate for competition submit",
        "config": gated["config"],
    }
    new_state = {**state, "inflight": inflight, "submission_kernel_created": True}
    return new_state, f"pushed submit_commit v{push.version}"


def _launch_eval(client: KaggleClient, state: dict) -> tuple[dict, str]:
    # Same single-holder rule as _launch_submit_commit (see comment there).
    if state.get("explore_inflight"):
        return state, "eval deferred: exploration commit holds the submission kernel"
    candidate = state["candidate"]
    kernel_id = submission_kernel_for(state)
    folder = stage_dir("eval")
    run_src = eval_run_src(candidate["config"])
    sources = dataset_sources_for(candidate["config"])
    write_submission_kernel(
        folder, run_src, dataset_sources=sources,
        model_sources=model_sources_for(candidate["config"]),
        kernel_id=kernel_id,
    )
    push = client.kernel_push(folder, accelerator=GPU_ACCELERATOR)
    if rotated := _rotate_burned_submission_kernel(state, push):
        return rotated
    if push.busy:
        return state, "eval push BUSY (2 GPU slots in use); retrying next tick"
    if not push.ok:
        return state, f"eval push FAILED: {push.error}"
    state = {**state, "submission_kernel_created": True}
    inflight = {
        "kernel": kernel_id,
        "version": push.version,
        "kind": "eval",
        "purpose": "measure trained candidate on the public-eval canary",
        "config": candidate["config"],
    }
    label = candidate.get("adapter_dataset", "no-adapter")
    return {**state, "inflight": inflight}, f"pushed eval v{push.version} for candidate ({label})"


# Consecutive push failures on ONE backlog item before it is skipped. Retrying
# (instead of the old skip-on-first-failure) is what keeps a transient push
# error from knocking a priority item out of line — the 2026-07-14 409 skipped
# the adapter train on its very first attempt. Three strikes still bounds a
# permanently-broken item so the loop can't wedge.
MAX_PUSH_FAILURES = 3


def _rotate_burned_submission_kernel(state: dict, push: PushResult) -> tuple[dict, str] | None:
    """Kaggle busy-create burn bug (2026-07-14, verified empirically): a push
    that would CREATE a new kernel id but is rejected on the GPU-session cap
    half-creates a broken record, and every later push to that id fails
    'Notebook not found'. When a submission-kernel push fails before the id
    ever had a successful push, rotate to the next `-N` id and retry next tick.
    Returns (new_state, msg) when rotation applies, else None. Never counts as
    a backlog strike — the failure isn't the item's fault."""
    if state.get("submission_kernel_created"):
        return None  # a successfully-created kernel updates fine; not a burn
    text = (push.error or "").lower()
    if not (push.busy or "notebook not found" in text):
        return None
    seq = int(state.get("submission_kernel_seq", 0)) + 1
    new_state = {**state, "submission_kernel_seq": seq}
    return new_state, (
        f"submission kernel id burned (create rejected — Kaggle busy-create bug); "
        f"rotated to {submission_kernel_for(new_state)}; retrying next tick"
    )


def _maybe_launch_backlog(client: KaggleClient, state: dict) -> tuple[dict, str]:
    quota = state.get("gpu_hours_week", 0.0)
    if quota >= WEEKLY_GPU_HOUR_QUOTA:
        msg = f"quota exhausted ({quota:.1f}h / {WEEKLY_GPU_HOUR_QUOTA:.0f}h this week); idle"
        return state, msg
    idx = state.get("backlog_index", 0) % len(BACKLOG)
    item = BACKLOG[idx]
    next_idx = (idx + 1) % len(BACKLOG)
    folder = item["build"](state)
    push = client.kernel_push(folder, accelerator=GPU_ACCELERATOR)
    if item["kind"] == "eval" and (rotated := _rotate_burned_submission_kernel(state, push)):
        # Burned submission-kernel id, not the item's fault: no strike, no skip.
        return rotated
    if push.busy:
        name = item["name"]
        return state, f"backlog[{name}] push BUSY (2 GPU slots in use); retrying next tick"
    if not push.ok:
        name = item["name"]
        failures = int(state.get("push_failures", 0)) + 1
        if failures >= MAX_PUSH_FAILURES:
            msg = f"backlog[{name}] push FAILED x{failures}; skipping item: {push.error}"
            return {**state, "backlog_index": next_idx, "push_failures": 0}, msg
        msg = (
            f"backlog[{name}] push FAILED ({failures}/{MAX_PUSH_FAILURES}); "
            f"retrying next tick: {push.error}"
        )
        return {**state, "push_failures": failures}, msg
    if item["kind"] == "eval":
        state = {**state, "submission_kernel_created": True}
        kernel = submission_kernel_for(state)
    else:
        kernel = TRAIN_KERNEL
    inflight = {
        "kernel": kernel,
        "version": push.version,
        "kind": item["kind"],
        "purpose": f"backlog:{item['name']}",
        "config": item["config"](state),
    }
    new_state = {
        **state,
        "inflight": inflight,
        "backlog_index": next_idx,
        "push_failures": 0,
        "gpu_hours_week": quota + item.get("estimated_hours", 1.0),
    }
    return new_state, f"launched backlog[{item['name']}] ({item['kind']}) v{push.version}"


# ---------------------------------------------------------------------------
# Exploration slot — a SECOND, narrow inflight lane (only ever a submit-commit
# on SUBMISSION_KERNEL) so an idle day's submission goes out even while a
# train/eval occupies the main slot. Kaggle allows 2 concurrent GPU sessions
# and the two lanes use different kernels; a push that still hits the session
# cap comes back `busy` and simply retries next tick.
# ---------------------------------------------------------------------------


def _dispatch_explore(client: KaggleClient, state: dict, today: str) -> tuple[dict, str | None]:
    explore = state.get("explore_inflight")
    if not explore:
        return state, None
    status = client.kernel_status(explore["kernel"]).upper()
    label = f"explore[{explore['variant']}] kernel {explore['kernel']} v{explore['version']}"
    if status == "COMPLETE":
        cleared = {**state, "explore_inflight": None}
        if state.get("last_submit_date") == today:
            # The gated path already used today's 1-submission slot; this
            # variant's ticket is spent (marked tried at launch) — never loop
            # a submit retry unattended.
            return cleared, f"{label} COMPLETE but today's submission slot already used"
        ref = f"{explore['kernel']}:v{explore['version']}:{today}"
        message = f"autopilot explore[{explore['variant']}] {ref}"
        if not client.competition_submit(explore["kernel"], explore["version"], message):
            return cleared, f"explore competition_submit FAILED for v{explore['version']}"
        new_state = {
            **cleared,
            "pending_lb": {"ref": ref, "config": explore["config"], "eval_correct": None},
            "last_submit_date": today,
        }
        return new_state, f"SUBMITTED (exploration): {ref}"
    if status in ("ERROR", "CANCEL"):
        return {**state, "explore_inflight": None}, f"FAILED: {label} status={status}"
    since = explore.get("since", today)
    if _days_between(since, today) >= STALE_INFLIGHT_DAYS:
        return {**state, "explore_inflight": None}, (
            f"STALE (>={STALE_INFLIGHT_DAYS}d unresolved) {label} status={status}; abandoning"
        )
    return (
        {**state, "explore_inflight": {**explore, "since": since}},
        f"waiting: {label} status={status}",
    )


def _maybe_launch_explore(
    client: KaggleClient, state: dict, today: str
) -> tuple[dict, str | None]:
    """Spend an otherwise-idle day's submission slot on the next UNTRIED
    exploration variant. Every guard is a reason to silently stand down —
    exploration must never displace the gated path, double-submit a day,
    clobber an unresolved pending_lb, or collide with a main-slot kernel."""
    if state.get("explore_inflight") is not None:
        return state, None
    if state.get("last_submit_date") == today:
        return state, None
    if state.get("pending_lb") is not None:
        return state, None  # one pending-LB record at a time (reconcile first)
    if state.get("gated_candidate") is not None:
        return state, None  # the gated path owns the next submission
    if state.get("candidate") is not None:
        return state, None  # the main lane is about to claim the submission kernel for eval
    inflight = state.get("inflight")
    if inflight and inflight.get("kind") in ("eval", "submit_commit"):
        return state, None  # collision guard: main slot holds the submission kernel
    quota = state.get("gpu_hours_week", 0.0)
    if quota + EXPLORE_COMMIT_HOURS > WEEKLY_GPU_HOUR_QUOTA:
        return state, None
    explored = set(state.get("explored", []))
    variant = next((v for v in exploration_variants(state) if v["name"] not in explored), None)
    if variant is None:
        return state, None  # rotation exhausted; a new adapter/candidate refills it
    kernel_id = submission_kernel_for(state)
    folder = stage_dir(f"explore_{variant['name']}")
    run_src = submission_run_src(variant["config"])
    write_submission_kernel(
        folder, run_src,
        dataset_sources=dataset_sources_for(variant["config"]),
        model_sources=model_sources_for(variant["config"]),
        kernel_id=kernel_id,
    )
    push = client.kernel_push(folder, accelerator=GPU_ACCELERATOR)
    if rotated := _rotate_burned_submission_kernel(state, push):
        return rotated
    if push.busy:
        return state, f"explore[{variant['name']}] push BUSY; retrying next tick"
    if not push.ok:
        return state, f"explore[{variant['name']}] push FAILED: {push.error}"
    explore_inflight = {
        "kernel": kernel_id,
        "version": push.version,
        "variant": variant["name"],
        "config": variant["config"],
    }
    new_state = {
        **state,
        "explore_inflight": explore_inflight,
        # Tried is marked at LAUNCH (not submit): an ERROR'd commit must not
        # re-enter the rotation forever in an unattended loop.
        "explored": sorted({*explored, variant["name"]}),
        "gpu_hours_week": quota + EXPLORE_COMMIT_HOURS,
        "submission_kernel_created": True,
    }
    return new_state, f"launched explore[{variant['name']}] commit v{push.version}"


def tick(client: KaggleClient, state: dict, today: str) -> dict:
    """Advance the state machine by exactly one step. `today` is an injected
    ISO date string (never `date.today()` internally) so this function is
    deterministic and side-effect-free apart from the `client` calls and the
    `SUBMISSION_LOG.md` append at the end."""
    state = _ensure_week_start(state, today)
    state, lb_msg = _reconcile_pending_lb(client, state, today)
    state, explore_msg = _dispatch_explore(client, state, today)

    if state.get("inflight"):
        state, msg = _dispatch_inflight(client, state, today)
    elif state.get("gated_candidate") and state.get("last_submit_date") != today:
        state, msg = _launch_submit_commit(client, state)
    elif state.get("candidate"):
        state, msg = _launch_eval(client, state)
    else:
        state, msg = _maybe_launch_backlog(client, state)

    # Exploration launch AFTER the main branch: it can run concurrently with a
    # train (different kernels; Kaggle allows 2 GPU sessions), and running it
    # last lets its collision guard see a SUBMISSION_KERNEL the main branch
    # claimed THIS tick.
    state, explore_launch_msg = _maybe_launch_explore(client, state, today)

    status = "; ".join(m for m in (lb_msg, explore_msg, msg, explore_launch_msg) if m)
    _log_status(today, status)
    return state


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _print_status() -> None:
    state = load_state()
    print(json.dumps(state, indent=2, sort_keys=True))
    if SUBMISSION_LOG_PATH.exists():
        lines = SUBMISSION_LOG_PATH.read_text(encoding="utf-8").splitlines()
        print("\n-- last SUBMISSION_LOG.md lines --")
        for line in lines[-10:]:
            print(line)


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="ARC-AGI-2 daily autopilot tick.")
    parser.add_argument(
        "--once", action="store_true", help="Run a single tick and exit (the only mode)."
    )
    parser.add_argument(
        "--status", action="store_true", help="Print current state + recent log, then exit."
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Use a KaggleClient that only logs and never shells out.",
    )
    return parser


def main(argv: list[str] | None = None) -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    args = _build_arg_parser().parse_args(argv)

    if args.status:
        _print_status()
        return

    client: KaggleClient = DryRunKaggleClient() if args.dry_run else KaggleClient()
    state = load_state()
    today = date.today().isoformat()
    state = tick(client, state, today)
    save_state(state)


if __name__ == "__main__":
    _scripts_dir = str(Path(__file__).resolve().parent)
    if _scripts_dir not in sys.path:
        sys.path.insert(0, _scripts_dir)
    main()
