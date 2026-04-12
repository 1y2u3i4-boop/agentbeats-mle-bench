"""MLE-Bench purple agent — AIDE tree search with AIRA_2 improvements.

AIRA_2 additions:
  - Parallel strategy attempts (asyncio.gather) instead of sequential loop
  - HCEEvaluator wired into each SolutionTree for noise-free scoring
"""
from __future__ import annotations

import asyncio
import base64
import io
import logging
import os
import shutil
import tarfile
import tempfile
from pathlib import Path

from a2a.server.tasks import TaskUpdater
from a2a.types import (
    FilePart,
    FileWithBytes,
    Message,
    Part,
    TaskState,
    TextPart,
)
from a2a.utils import new_agent_text_message

from hce import HCEEvaluator
from llm import LLMClient
from strategies import get_strategy, all_strategy_names
from tree import SolutionTree

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "")
OPENAI_MODEL = os.environ.get("OPENAI_MODEL", "o4-mini")
MAX_ITERATIONS = int(os.environ.get("MAX_ITERATIONS", "12"))
CODE_TIMEOUT = int(os.environ.get("CODE_TIMEOUT", "600"))
NUM_ATTEMPTS = int(os.environ.get("NUM_ATTEMPTS", "3"))


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _extract_tar_b64(b64_text: str, dest: Path) -> None:
    raw = base64.b64decode(b64_text)
    dest.mkdir(parents=True, exist_ok=True)
    with tarfile.open(fileobj=io.BytesIO(raw), mode="r:gz") as tar:
        tar.extractall(dest, filter="data")


def _first_tar_from_message(message: Message) -> str | None:
    for part in message.parts:
        root = part.root
        if isinstance(root, FilePart):
            fd = root.file
            if isinstance(fd, FileWithBytes) and fd.bytes is not None:
                raw = fd.bytes
                if isinstance(raw, str):
                    return raw
                if isinstance(raw, (bytes, bytearray)):
                    return base64.b64encode(raw).decode("ascii")
    return None


# ---------------------------------------------------------------------------
# Agent
# ---------------------------------------------------------------------------

class Agent:
    def __init__(self):
        self._done_contexts: set[str] = set()

    async def run(self, message: Message, updater: TaskUpdater) -> None:
        ctx = message.context_id or "default"
        if ctx in self._done_contexts:
            return

        tar_b64 = _first_tar_from_message(message)
        if not tar_b64:
            await updater.add_artifact(
                parts=[Part(root=TextPart(text="Error: no competition tar.gz in message"))],
                name="Error",
            )
            return

        api_key = OPENAI_API_KEY
        if not api_key:
            await updater.add_artifact(
                parts=[Part(root=TextPart(text="Error: OPENAI_API_KEY env var required"))],
                name="Error",
            )
            return

        strategy_names = all_strategy_names()[:NUM_ATTEMPTS]
        if len(strategy_names) < NUM_ATTEMPTS:
            strategy_names = (strategy_names * (NUM_ATTEMPTS // len(strategy_names) + 1))[:NUM_ATTEMPTS]

        await updater.update_status(
            TaskState.working,
            new_agent_text_message(
                f"Starting MLE-Bench solve: {NUM_ATTEMPTS} parallel attempt(s), "
                f"model={OPENAI_MODEL}, iterations={MAX_ITERATIONS}, HCE=enabled"
            ),
        )

        loop = asyncio.get_running_loop()

        # AIRA_2: launch all strategy attempts in parallel
        attempt_tasks = [
            self._run_attempt(i, strat, tar_b64, api_key, updater, loop)
            for i, strat in enumerate(strategy_names)
        ]
        attempt_results = await asyncio.gather(*attempt_tasks, return_exceptions=True)

        # Pick the best result across all parallel attempts
        best_result = None
        best_score = None
        best_csv_bytes = None

        for outcome in attempt_results:
            if isinstance(outcome, Exception):
                logger.error("Attempt raised exception: %s", outcome)
                continue
            result, csv_bytes, score, strat_name = outcome
            if csv_bytes is None:
                continue
            if best_csv_bytes is None or (score is not None and (best_score is None or score > best_score)):
                best_result = result
                best_score = score
                best_csv_bytes = csv_bytes
                logger.info("New best from strategy=%s score=%s", strat_name, score)

        if best_csv_bytes is None:
            await updater.add_artifact(
                parts=[Part(root=TextPart(text="Error: no valid submission produced across all attempts"))],
                name="Error",
            )
            return

        b64_out = base64.b64encode(best_csv_bytes).decode("ascii")
        summary = (
            f"Complete: {NUM_ATTEMPTS} parallel attempt(s), "
            f"best_score={best_score}, "
            f"nodes={len(best_result.all_nodes) if best_result else 0}, "
            f"model={OPENAI_MODEL}"
        )

        await updater.add_artifact(
            parts=[
                Part(root=TextPart(text=summary)),
                Part(root=FilePart(file=FileWithBytes(
                    bytes=b64_out,
                    name="submission.csv",
                    mime_type="text/csv",
                ))),
            ],
            name="submission",
        )
        self._done_contexts.add(ctx)
        logger.info("Submitted: %d bytes, %s", len(best_csv_bytes), summary)

    async def _run_attempt(
        self,
        attempt_idx: int,
        strat_name: str,
        tar_b64: str,
        api_key: str,
        updater: TaskUpdater,
        loop: asyncio.AbstractEventLoop,
    ) -> tuple:
        """Run one strategy attempt in a thread; return (result, csv_bytes, score, strat_name)."""
        tmpdir = tempfile.mkdtemp(prefix=f"mle-{attempt_idx}-")
        try:
            workdir = Path(tmpdir)
            _extract_tar_b64(tar_b64, workdir)

            # AIRA_2: set up HCE hidden validation split
            hce = HCEEvaluator()
            hce_active = hce.setup(workdir)

            await updater.update_status(
                TaskState.working,
                new_agent_text_message(
                    f"Attempt {attempt_idx + 1}/{NUM_ATTEMPTS}: strategy={strat_name}, HCE={'on' if hce_active else 'off'}"
                ),
            )

            llm = LLMClient(api_key=api_key, model=OPENAI_MODEL)
            strategy_text = get_strategy(strat_name)
            tree = SolutionTree(
                workdir=workdir,
                llm=llm,
                max_iterations=MAX_ITERATIONS,
                code_timeout=CODE_TIMEOUT,
                strategy_name=strat_name,
                hce=hce if hce_active else None,
            )

            def make_callback(sname: str):
                def on_node_complete(node):
                    score_str = ""
                    if node.hce_score is not None:
                        score_str = f"hce={node.hce_score:.4f}"
                    elif node.cv_score is not None:
                        score_str = f"cv={node.cv_score:.4f}"
                    try:
                        asyncio.run_coroutine_threadsafe(
                            updater.update_status(
                                TaskState.working,
                                new_agent_text_message(
                                    f"[{sname}] Node {node.node_id}: {score_str} "
                                    f"err={node.error} t={node.exec_time:.0f}s"
                                ),
                            ),
                            loop,
                        )
                    except Exception:
                        pass
                return on_node_complete

            result = await loop.run_in_executor(
                None,
                lambda t=tree, s=strategy_text, cb=make_callback(strat_name): t.run(s, on_node_complete=cb),
            )

            submission_path = workdir / "submission.csv"
            csv_bytes = submission_path.read_bytes() if submission_path.exists() else None

            if result.best_node is not None:
                # Prefer HCE score as the ranking signal (AIRA_2)
                score = result.best_node.hce_score if result.best_node.hce_score is not None else result.best_node.cv_score
            else:
                score = None

            return result, csv_bytes, score, strat_name

        except Exception as exc:
            logger.exception("Attempt %d (%s) failed: %s", attempt_idx, strat_name, exc)
            return None, None, None, strat_name
        finally:
            shutil.rmtree(tmpdir, ignore_errors=True)
