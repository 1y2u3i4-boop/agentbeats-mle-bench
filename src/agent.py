"""MLE-Bench purple agent: receives competition data, solves it using AIDE-style tree search."""

from __future__ import annotations

import base64
import io
import logging
import os
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
from a2a.utils import get_message_text, new_agent_text_message

from llm import LLMClient
from strategies import DEFAULT_STRATEGY, get_strategy, all_strategy_names
from tree import SolutionTree

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "")
OPENAI_MODEL = os.environ.get("OPENAI_MODEL", "o4-mini")
MAX_ITERATIONS = int(os.environ.get("MAX_ITERATIONS", "12"))
CODE_TIMEOUT = int(os.environ.get("CODE_TIMEOUT", "600"))
NUM_ATTEMPTS = int(os.environ.get("NUM_ATTEMPTS", "3"))


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


class Agent:
    def __init__(self):
        self._done_contexts: set[str] = set()

    async def run(self, message: Message, updater: TaskUpdater) -> None:
        ctx = message.context_id or "default"
        if ctx in self._done_contexts:
            return

        # Extract competition tar from message
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

        # Determine strategies for multiple attempts
        strategy_names = all_strategy_names()[:NUM_ATTEMPTS]
        if len(strategy_names) < NUM_ATTEMPTS:
            strategy_names = strategy_names * (NUM_ATTEMPTS // len(strategy_names) + 1)
            strategy_names = strategy_names[:NUM_ATTEMPTS]

        await updater.update_status(
            TaskState.working,
            new_agent_text_message(
                f"Starting MLE-Bench solve with {NUM_ATTEMPTS} attempt(s), "
                f"model={OPENAI_MODEL}, iterations={MAX_ITERATIONS}"
            ),
        )

        best_result = None
        best_score = None
        best_csv_bytes = None

        for attempt_idx, strat_name in enumerate(strategy_names):
            strategy_text = get_strategy(strat_name)

            await updater.update_status(
                TaskState.working,
                new_agent_text_message(
                    f"Attempt {attempt_idx + 1}/{NUM_ATTEMPTS}: strategy={strat_name}"
                ),
            )

            with tempfile.TemporaryDirectory(prefix=f"mle-{ctx}-{attempt_idx}-") as tmpdir:
                workdir = Path(tmpdir)
                try:
                    _extract_tar_b64(tar_b64, workdir)
                except Exception as exc:
                    logger.error("Failed to extract tar for attempt %d: %s", attempt_idx, exc)
                    continue

                llm = LLMClient(api_key=api_key, model=OPENAI_MODEL)
                tree = SolutionTree(
                    workdir=workdir,
                    llm=llm,
                    max_iterations=MAX_ITERATIONS,
                    code_timeout=CODE_TIMEOUT,
                    strategy_name=strat_name,
                )

                loop = __import__("asyncio").get_running_loop()

                def make_callback(sname):
                    def on_node_complete(node):
                        try:
                            __import__("asyncio").run_coroutine_threadsafe(
                                updater.update_status(
                                    TaskState.working,
                                    new_agent_text_message(
                                        f"[{sname}] Node {node.node_id}: "
                                        f"cv={node.cv_score} err={node.error} "
                                        f"time={node.exec_time:.0f}s"
                                    ),
                                ),
                                loop,
                            )
                        except Exception:
                            pass
                    return on_node_complete

                try:
                    result = await loop.run_in_executor(
                        None,
                        lambda t=tree, s=strategy_text, cb=make_callback(strat_name): t.run(s, on_node_complete=cb),
                    )
                except Exception as exc:
                    logger.exception("Tree search failed for attempt %d", attempt_idx)
                    continue

                # Check if this attempt produced a better result
                submission_path = workdir / "submission.csv"
                if submission_path.exists():
                    csv_bytes = submission_path.read_bytes()
                    score = result.best_node.cv_score if result.best_node else None

                    if best_csv_bytes is None or (
                        score is not None and (best_score is None or score > best_score)
                    ):
                        best_result = result
                        best_score = score
                        best_csv_bytes = csv_bytes
                        logger.info(
                            "Attempt %d improved best: score=%s strategy=%s",
                            attempt_idx, score, strat_name,
                        )

        # Return best submission
        if best_csv_bytes is None:
            await updater.add_artifact(
                parts=[Part(root=TextPart(text="Error: no valid submission produced across all attempts"))],
                name="Error",
            )
            return

        b64_out = base64.b64encode(best_csv_bytes).decode("ascii")
        summary = (
            f"Complete: {NUM_ATTEMPTS} attempt(s), "
            f"best_score={best_score}, "
            f"nodes={len(best_result.all_nodes) if best_result else 0}, "
            f"model={OPENAI_MODEL}"
        )

        await updater.add_artifact(
            parts=[
                Part(root=TextPart(text=summary)),
                Part(
                    root=FilePart(
                        file=FileWithBytes(
                            bytes=b64_out,
                            name="submission.csv",
                            mime_type="text/csv",
                        )
                    )
                ),
            ],
            name="submission",
        )
        self._done_contexts.add(ctx)
        logger.info("Submitted: %d bytes, %s", len(best_csv_bytes), summary)
