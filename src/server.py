import argparse
import uvicorn

from a2a.server.apps import A2AStarletteApplication
from a2a.server.request_handlers import DefaultRequestHandler
from a2a.server.tasks import InMemoryTaskStore
from a2a.types import (
    AgentCapabilities,
    AgentCard,
    AgentSkill,
)

from executor import Executor


def main():
    parser = argparse.ArgumentParser(description="Run the MLE-Bench purple agent.")
    parser.add_argument("--host", type=str, default="127.0.0.1", help="Host to bind the server")
    parser.add_argument("--port", type=int, default=9009, help="Port to bind the server")
    parser.add_argument("--card-url", type=str, help="URL to advertise in the agent card")
    args = parser.parse_args()

    skill = AgentSkill(
        id="mle-bench-solver",
        name="MLE-Bench Competition Solver",
        description=(
            "Solves Kaggle-style ML competitions from MLE-Bench. "
            "Receives competition data as a tar.gz archive, runs AIDE-style tree search "
            "to iteratively generate and improve ML solutions, and returns submission.csv."
        ),
        tags=["mle-bench", "kaggle", "machine-learning", "competition"],
        examples=[],
    )

    agent_card = AgentCard(
        name="MLE-Bench Purple Agent",
        description=(
            "An autonomous ML engineering agent that solves Kaggle competitions "
            "using AIDE-style tree search with iterative code generation and execution."
        ),
        url=args.card_url or f"http://{args.host}:{args.port}/",
        version="1.0.0",
        default_input_modes=["text", "application/gzip"],
        default_output_modes=["text", "text/csv"],
        capabilities=AgentCapabilities(streaming=True),
        skills=[skill],
    )

    request_handler = DefaultRequestHandler(
        agent_executor=Executor(),
        task_store=InMemoryTaskStore(),
    )
    server = A2AStarletteApplication(
        agent_card=agent_card,
        http_handler=request_handler,
        max_content_length=None,
    )
    uvicorn.run(server.build(), host=args.host, port=args.port)


if __name__ == "__main__":
    main()
